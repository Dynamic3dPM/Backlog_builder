import { createRouter, createWebHistory } from 'vue-router';
import GitHubIntegration from './views/GitHubIntegration.vue';
import Upload from './views/Upload.vue';

const routes = [
  {
    path: '/upload',
    name: 'Upload',
    component: Upload
  },
  {
    path: '/github',
    name: 'GitHubIntegration',
    component: GitHubIntegration
  },
  {
    path: '/',
    redirect: '/upload'
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

export default router;
