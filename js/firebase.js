import { initializeApp } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-app.js";
import { getDatabase, ref, onValue, push, set, get, query, limitToLast } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-database.js";

const firebaseConfig = {
  apiKey: "AIzaSyCkA910fAJd2CbLLU3JzXI1ff2Xw4WM9Zs",
  authDomain: "co2-monitor-effff.firebaseapp.com",
  databaseURL: "https://co2-monitor-effff-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "co2-monitor-effff",
  storageBucket: "co2-monitor-effff.firebasestorage.app",
  messagingSenderId: "1045222550408",
  appId:"1:1045222550408:web:b9401d197d613b37de683d"
};

const app = initializeApp(firebaseConfig);
const db = getDatabase(app);

export { db, ref, onValue, push, set, get, query, limitToLast };