import { db, ref, onValue, set, get } from "./firebase.js";
import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js";

const MODEL_URL = "https://mahtoyash.github.io/co2-monitor/model/model.json";
const SEQ_LEN = 30;
const FEATURE_MIN = [18.04, 44.92, 415.0, 0.0, 0.0, 0.0, 0.0];
const FEATURE_MAX = [25.57, 58.15, 1400.0, 60.0, 23.0, 6.0, 59.0];
const TARGET_MIN = 415.0;
const TARGET_MAX = 1400.0;

let currentRoom = "room1";
let roomModel = null;
let prevCO2 = null;
let co2History = [];
let newSamplesBuffer = [];

function scaleFeatures(row) {
  return row.map((val, i) => (val - FEATURE_MIN[i]) / (FEATURE_MAX[i] - FEATURE_MIN[i]));
}

function unscaleTarget(val) {
  return val * (TARGET_MAX - TARGET_MIN) + TARGET_MIN;
}

async function loadRoomModel(roomId) {
  const snapshot = await get(ref(db, `model_weights/${roomId}`));
  if (snapshot.exists()) {
    console.log(`Saved weights found for ${roomId}`);
    roomModel = await tf.loadLayersModel(MODEL_URL);
    const weightsData = snapshot.val();
    const weightTensors = weightsData.map(w => tf.tensor(w.data, w.shape));
    roomModel.setWeights(weightTensors);
  } else {
    console.log(`No saved weights for ${roomId}, using base model`);
    roomModel = await tf.loadLayersModel(MODEL_URL);
  }
  console.log("Model ready!");
}

async function saveRoomWeights(roomId) {
  const weights = roomModel.getWeights();
  const weightsData = await Promise.all(
    weights.map(async (w) => ({
      data: Array.from(await w.data()),
      shape: w.shape
    }))
  );
  await set(ref(db, `model_weights/${roomId}`), weightsData);
  console.log(`Weights saved for ${roomId}`);
}

async function runPrediction(history) {
  if (!roomModel || history.length < SEQ_LEN) return;
  const last30 = history.slice(-SEQ_LEN);
  const scaled = last30.map(r => scaleFeatures([
    r.temperature, r.humidity, r.co2,
    r.occupancy || 0, r.hour, r.day_of_week, r.minute
  ]));
  const input = tf.tensor3d([scaled]);
  const output = roomModel.predict(input);
  const preds = Array.from(await output.data());
  document.getElementById("pred-15").textContent = Math.round(unscaleTarget(preds[0])) + " ppm";
  document.getElementById("pred-45").textContent = Math.round(unscaleTarget(preds[1])) + " ppm";
  document.getElementById("pred-60").textContent = Math.round(unscaleTarget(preds[2])) + " ppm";
  input.dispose();
  output.dispose();
}

async function onlineTrain(history) {
  if (!roomModel || history.length < SEQ_LEN + 1) return;
  const last31 = history.slice(-(SEQ_LEN + 1));
  const inputSeq = last31.slice(0, SEQ_LEN);
  const target = last31[SEQ_LEN];
  const scaled = inputSeq.map(r => scaleFeatures([
    r.temperature, r.humidity, r.co2,
    r.occupancy || 0, r.hour, r.day_of_week, r.minute
  ]));
  const targetScaled = (target.co2 - TARGET_MIN) / (TARGET_MAX - TARGET_MIN);
  const xs = tf.tensor3d([scaled]);
  const ys = tf.tensor2d([[targetScaled, targetScaled, targetScaled]]);
  roomModel.compile({ optimizer: tf.train.adam(0.001), loss: "meanSquaredError" });
  await roomModel.fit(xs, ys, { epochs: 1, verbose: 0 });
  xs.dispose();
  ys.dispose();
  newSamplesBuffer.push(1);
  if (newSamplesBuffer.length >= 10) {
    await saveRoomWeights(currentRoom);
    newSamplesBuffer = [];
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
      tension: 0.4,
      fill: false
    }]
  },
  options: {
    responsive: true,
    scales: { y: { beginAtZero: false } }
  }
});

function updateChart() {
  const last20 = co2History.slice(-20);
  chart.data.labels = last20.map((_, i) => i + 1);
  chart.data.datasets[0].data = last20.map(r => r.co2);
  chart.update();
}

function listenToRoom(roomId) {
  onValue(ref(db, `rooms/${roomId}/latest`), (snapshot) => {
    const data = snapshot.val();
    if (!data) return;
    document.getElementById("co2-value").textContent  = data.co2 + " ppm";
    document.getElementById("temp-value").textContent = data.temperature + " °C";
    document.getElementById("hum-value").textContent  = data.humidity + " %";
    if (prevCO2 !== null && (data.co2 - prevCO2) >= 200) {
      document.getElementById("spike-alert").style.display = "block";
    } else {
      document.getElementById("spike-alert").style.display = "none";
    }
    prevCO2 = data.co2;
  });

  onValue(ref(db, `rooms/${roomId}/history`), async (snapshot) => {
    co2History = [];
    snapshot.forEach(child => co2History.push(child.val()));
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
    await loadRoomModel(currentRoom);
    listenToRoom(currentRoom);
  });

  document.getElementById("occupancy-btn").addEventListener("click", () => {
    const val = parseInt(document.getElementById("occupancy-input").value);
    set(ref(db, `rooms/${currentRoom}/latest/occupancy`), val);
  });
}

init();