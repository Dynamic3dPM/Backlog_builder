import { createApp } from 'vue';
import App from './App.vue';
import router from './router';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap';

// Dynamically load Bootstrap Icons CSS
const link = document.createElement('link');
link.href = 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css';
link.rel = 'stylesheet';
document.head.appendChild(link);

import Toast from 'vue3-toastify';
import 'vue3-toastify/dist/index.css';
import './assets/styles/main.css';

const app = createApp(App);

// Configure toast options
const toastOptions = {
  position: 'top-right',
  autoClose: 5000,
  hideProgressBar: false,
  closeOnClick: true,
  pauseOnHover: true,
  draggable: true,
  progress: undefined,
};

// Add plugins
app.use(router);
app.use(Toast, toastOptions);

// Add axios to the global properties
app.config.globalProperties.$http = axios;

app.mount('#app');