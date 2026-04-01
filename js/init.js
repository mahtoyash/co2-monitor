// This file forcefully disables WebSockets in the browser to make Firebase Realtime Database
// instantly fallback to HTTP Long-Polling. This completely cures the 1-minute 24-second
// timeout delay on restricted Wi-Fi networks where WebSockets are blocked.

window.WebSocket = undefined;
console.log("WebSocket explicitly disabled. Forcing instant Firebase HTTP Long-Polling.");
