import { createRouter, createWebHistory } from 'vue-router';
import GitHubIntegration from './views/GitHubIntegration.vue';
import Upload from './views/Upload.vue';
import BacklogWorkflow from './views/BacklogWorkflow.vue';

const routes = [
  {
    path: '/workflow',
    name: 'BacklogWorkflow',
    component: BacklogWorkflow
  },
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
    redirect: '/workflow'
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

export default router;
