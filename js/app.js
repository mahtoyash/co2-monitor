import { db, ref, onValue } from "./firebase.js";

let currentRoom = "room1";
let prevCO2 = null;
let co2History = [];

// ── Room switch ──────────────────────────────────
document.getElementById("room-select").addEventListener("change", (e) => {
  currentRoom = e.target.value;
  listenToRoom(currentRoom);
});

// ── Listen to Firebase ───────────────────────────
function listenToRoom(roomId) {
  const latestRef = ref(db, `rooms/${roomId}/latest`);
  onValue(latestRef, (snapshot) => {
    const data = snapshot.val();
    if (!data) return;

    // Update UI
    document.getElementById("co2-value").textContent  = data.co2 + " ppm";
    document.getElementById("temp-value").textContent = data.temperature + " °C";
    document.getElementById("hum-value").textContent  = data.humidity + " %";

    // Spike check
    if (prevCO2 !== null && (data.co2 - prevCO2) >= 200) {
      document.getElementById("spike-alert").style.display = "block";
    } else {
      document.getElementById("spike-alert").style.display = "none";
    }
    prevCO2 = data.co2;

    // Chart update
    co2History.push(data.co2);
    if (co2History.length > 20) co2History.shift(); // last 20 readings
    updateChart();
  });
}

// ── Chart.js ─────────────────────────────────────
const ctx = document.getElementById("co2-chart").getContext("2d");
const chart = new Chart(ctx, {
  type: "line",
  data: {
    labels: co2History.map((_, i) => i + 1),
    datasets: [{
      label: "CO2 (ppm)",
      data: co2History,
      borderColor: "#4CAF50",
      tension: 0.4,
      fill: false
    }]
  },
  options: {
    responsive: true,
    scales: {
      y: { beginAtZero: false }
    }
  }
});

function updateChart() {
  chart.data.labels = co2History.map((_, i) => i + 1);
  chart.data.datasets[0].data = co2History;
  chart.update();
}

// ── Start listening ───────────────────────────────
listenToRoom(currentRoom);