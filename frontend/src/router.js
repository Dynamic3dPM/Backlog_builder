import { createRouter, createWebHistory } from 'vue-router';
import GitHubIntegration from './views/GitHubIntegration.vue';

const routes = [
  {
    path: '/github',
    name: 'GitHubIntegration',
    component: GitHubIntegration
  },
  // Add more routes as needed
  {
    path: '/',
    redirect: '/github'
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

export default router;
