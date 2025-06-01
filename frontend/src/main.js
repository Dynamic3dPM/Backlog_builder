import { createApp } from 'vue';
import App from './App.vue';
import router from './router';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap';
import Vue3Toastify from 'vue3-toastify';
import 'vue3-toastify/dist/index.css';
import './assets/styles/main.css';

const app = createApp(App);

// Configure toast options
const toastOptions = {
  position: 'top-right',
  timeout: 5000,
  closeOnClick: true,
  pauseOnFocusLoss: true,
  pauseOnHover: true,
  draggable: true,
  draggablePercent: 60,
  hideProgressBar: false,
  icon: true,
};

// Add plugins
app.use(router);
app.use(Vue3Toastify, toastOptions);

app.mount('#app');