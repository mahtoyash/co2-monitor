import { db, ref, onValue, set, get, query, limitToLast } from "./firebase.js";

const BASE_WEIGHTS_URL = "model/base_weights.json";
const SEQ_LEN = 24;
const FEATURE_MIN = [11.6, 21.8, 0.0, 400.0, 0.0, 0.0, 0.0];
const FEATURE_MAX = [25.6, 80.9, 29.0, 1368.0, 23.0, 6.0, 59.0];
const TARGET_MIN = 400.0;
const TARGET_MAX = 1368.0;

let currentRoom = "room1";
let roomModel = null;
let prevCO2 = null;
let co2History = [];
let newSamplesBuffer = [];
let unsubLatest = null;
let unsubHistory = null;

function buildModel() {
  const model = tf.sequential();
  model.add(tf.layers.lstm({ units: 128, returnSequences: true, inputShape: [24, 7] }));
  model.add(tf.layers.lstm({ units: 128, returnSequences: false }));
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 3 }));
  model.compile({ optimizer: tf.train.adam(0.001), loss: "meanSquaredError" });
  model.predict(tf.zeros([1, 24, 7]));
  return model;
}

function scaleFeatures(row) {
  return row.map((val, i) => (val - FEATURE_MIN[i]) / (FEATURE_MAX[i] - FEATURE_MIN[i]));
}

function unscaleTarget(val) {
  return val * (TARGET_MAX - TARGET_MIN) + TARGET_MIN;
}

async function loadRoomModel(roomId) {
  try {
    roomModel = buildModel();
    const snapshot = await get(ref(db, `model_weights/${roomId}`));
    if (snapshot.exists()) {
      console.log(`Saved weights found for ${roomId}, applying...`);
      const weightsData = snapshot.val();
      const weightTensors = weightsData.map(w => tf.tensor(w.data, w.shape));
      roomModel.setWeights(weightTensors);
      console.log("Per-room weights applied ✓");
    } else {
      console.log(`No saved weights for ${roomId}, loading base weights...`);
      const resp = await fetch(BASE_WEIGHTS_URL);
      const weightsData = await resp.json();
      const weightTensors = weightsData.map(w => tf.tensor(w));
      roomModel.setWeights(weightTensors);
      console.log("Base weights applied ✓");
    }
    console.log("Model ready!");
  } catch (err) {
    console.error("Model load failed:", err);
  }
}

async function saveRoomWeights(roomId) {
  try {
    const weights = roomModel.getWeights();
    const weightsData = await Promise.all(
      weights.map(async (w) => ({
        data: Array.from(await w.data()),
        shape: w.shape
      }))
    );
    await set(ref(db, `model_weights/${roomId}`), weightsData);
    console.log(`Weights saved for ${roomId} ✓`);
  } catch (err) {
    console.error("Weight save failed:", err);
  }
}

async function runPrediction(history) {
  if (!roomModel || history.length < SEQ_LEN) {
    console.log(`Not enough history: ${history.length}/${SEQ_LEN}`);
    return;
  }
  try {
    const lastSeq = history.slice(-SEQ_LEN);
    const scaled = lastSeq.map(r => scaleFeatures([
      r.temperature, r.humidity, r.occupancy !== undefined ? r.occupancy : 0, r.co2,
      r.hour, r.day_of_week, r.minute
    ]));
    const input = tf.tensor3d([scaled], [1, SEQ_LEN, 7]);
    const output = roomModel.predict(input);
    const preds = Array.from(await output.data());
    document.getElementById("pred-10").textContent = Math.round(unscaleTarget(preds[0])) + " ppm";
    document.getElementById("pred-30").textContent = Math.round(unscaleTarget(preds[1])) + " ppm";
    document.getElementById("pred-60").textContent = Math.round(unscaleTarget(preds[2])) + " ppm";
    input.dispose();
    output.dispose();
  } catch (err) {
    console.error("Prediction failed:", err);
  }
}

async function onlineTrain(history) {
  // We need predicting targets for +10, +30, +60 minutes.
  // Including the 24 min sequence length + 61 steps mapping = 85
  if (!roomModel || history.length < 85) return;
  try {
    const seqEnd = history.length - 61;
    const seqStart = seqEnd - SEQ_LEN; // 24 elements length
    const inputSeq = history.slice(seqStart, seqEnd);

    const target10 = history[seqEnd + 10];
    const target30 = history[seqEnd + 30];
    const target60 = history[seqEnd + 60];

    const scaled = inputSeq.map(r => scaleFeatures([
      r.temperature, r.humidity, r.occupancy !== undefined ? r.occupancy : 0, r.co2,
      r.hour, r.day_of_week, r.minute
    ]));
    
    const t10Scaled = (target10.co2 - TARGET_MIN) / (TARGET_MAX - TARGET_MIN);
    const t30Scaled = (target30.co2 - TARGET_MIN) / (TARGET_MAX - TARGET_MIN);
    const t60Scaled = (target60.co2 - TARGET_MIN) / (TARGET_MAX - TARGET_MIN);

    const xs = tf.tensor3d([scaled], [1, SEQ_LEN, 7]);
    const ys = tf.tensor2d([[t10Scaled, t30Scaled, t60Scaled]]);
    
    await roomModel.fit(xs, ys, { epochs: 1, verbose: 0 });
    xs.dispose();
    ys.dispose();
    newSamplesBuffer.push(1);
    if (newSamplesBuffer.length >= 10) {
      await saveRoomWeights(currentRoom);
      newSamplesBuffer = [];
    }
  } catch (err) {
    console.error("Online training failed:", err);
  }
}

const ctx = document.getElementById("co2-chart").getContext("2d");
const chart = new Chart(ctx, {
  type: "line",
  data: {
    labels: [],
    datasets: [{
      label: "CO2 (ppm)",
      data: [],
      borderColor: "#4CAF50",
      backgroundColor: "rgba(76, 175, 80, 0.1)",
      tension: 0.4,
      fill: true,
      pointRadius: 3
    }]
  },
  options: {
    responsive: true,
    plugins: { legend: { display: true } },
    scales: {
      y: { beginAtZero: false, title: { display: true, text: "CO2 (ppm)" } },
      x: { title: { display: true, text: "Reading #" } }
    }
  }
});

function updateChart() {
  const last20 = co2History.slice(-20);
  chart.data.labels = last20.map((_, i) => i + 1);
  chart.data.datasets[0].data = last20.map(r => r.co2);
  chart.update();
}

function listenToRoom(roomId) {
  // Purane listeners unsubscribe karo
  if (unsubLatest) unsubLatest();
  if (unsubHistory) unsubHistory();

  unsubLatest = onValue(ref(db, `rooms/${roomId}/latest`), (snapshot) => {
    const data = snapshot.val();
    if (!data) return;
    document.getElementById("co2-value").textContent  = data.co2 + " ppm";
    document.getElementById("temp-value").textContent = data.temperature + " °C";
    document.getElementById("hum-value").textContent  = data.humidity + " %";
    document.getElementById("occ-value").textContent  = (data.occupancy || 0);
    if (prevCO2 !== null && (data.co2 - prevCO2) >= 200) {
      document.getElementById("spike-alert").style.display = "block";
    } else {
      document.getElementById("spike-alert").style.display = "none";
    }
    prevCO2 = data.co2;
  });

  const historyQuery = query(ref(db, `rooms/${roomId}/history`), limitToLast(90));
  unsubHistory = onValue(historyQuery, async (snapshot) => {
    co2History = [];
    snapshot.forEach((childSnap) => {
      co2History.push(childSnap.val());
    });
    console.log(`History loaded: ${co2History.length} readings`);
    updateChart();
    await runPrediction(co2History);
    await onlineTrain(co2History);
  });
}

async function init() {
  await loadRoomModel(currentRoom);
  listenToRoom(currentRoom);

  document.getElementById("room-select").addEventListener("change", async (e) => {
    currentRoom = e.target.value;
    newSamplesBuffer = [];
    prevCO2 = null;
    co2History = [];
    await loadRoomModel(currentRoom);
    listenToRoom(currentRoom);
  });

  document.getElementById("occupancy-btn").addEventListener("click", async () => {
    const val = parseInt(document.getElementById("occupancy-input").value) || 0;
    
    // 1. Update the latest display
    await set(ref(db, `rooms/${currentRoom}/latest/occupancy`), val);
    
    // 2. Override the last item in the history loop so predictions update instantly
    const snap = await get(query(ref(db, `rooms/${currentRoom}/history`), limitToLast(1)));
    if (snap.exists()) {
      const histObj = snap.val();
      const keys = Object.keys(histObj);
      if (keys.length > 0) {
        const lastKey = keys[keys.length - 1];
        await set(ref(db, `rooms/${currentRoom}/history/${lastKey}/occupancy`), val);
      }
    }

    // Give a tiny flash feedback to the button
    const btn = document.getElementById("occupancy-btn");
    const oldText = btn.textContent;
    btn.textContent = "Saved & Predicting!";
    setTimeout(() => { btn.textContent = oldText; }, 1500);
  });
}

init();

