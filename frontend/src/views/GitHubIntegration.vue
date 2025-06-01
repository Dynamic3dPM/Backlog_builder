<template>
  <div class="github-integration">
    <h2>GitHub Integration</h2>
    
    <div class="card">
      <div class="card-body">
        <h3 class="card-title">Create GitHub Issue</h3>
        
        <ProjectSelector 
          v-model="selection"
          @change="onSelectionChange"
          @error="onError"
        />
        
        <div v-if="selection.project && selection.repository" class="mt-4">
          <button 
            class="btn btn-primary"
            @click="createSampleIssue"
            :disabled="creatingIssue"
          >
            {{ creatingIssue ? 'Creating...' : 'Create Sample Issue' }}
          </button>
          
          <div v-if="createdIssue" class="mt-3 alert alert-success">
            <h4>Issue Created Successfully!</h4>
            <p>Issue URL: <a :href="createdIssue.issue.url" target="_blank">{{ createdIssue.issue.url }}</a></p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { toast } from 'vue3-toastify';
import 'vue3-toastify/dist/index.css';
import ProjectSelector from '@/components/ProjectSelector.vue';
import githubService from '@/services/github';

export default {
  name: 'GitHubIntegration',
  components: {
    ProjectSelector
  },
  setup() {
    return { toast }
  },
  data() {
    return {
      selection: {
        project: null,
        repository: null
      },
      creatingIssue: false,
      createdIssue: null
    };
  },
  methods: {
    onSelectionChange({ project, repository }) {
      console.log('Selection changed:', { project, repository });
      this.selection = { project, repository };
      this.createdIssue = null; // Reset created issue when selection changes
    },
    
    onError(message) {
      this.toast.error(message, { position: 'top-right' });
    },
    
    async createSampleIssue() {
      if (!this.selection.repository) {
        this.onError('Please select a repository');
        return;
      }
      
      const sampleIssue = {
        title: 'Sample Issue from Backlog Builder',
        description: 'This is a sample issue created by Backlog Builder.',
        type: 'Task',
        priority: 'Medium'
      };
      
      this.creatingIssue = true;
      
      try {
        this.createdIssue = await githubService.createIssue(
          sampleIssue,
          this.selection.repository.name,
          this.selection.project?.id
        );
        this.toast.success('Issue created successfully!', { position: 'top-right' });
      } catch (error) {
        console.error('Failed to create issue:', error);
        this.toast.error('Failed to create issue', { position: 'top-right' });
      } finally {
        this.creatingIssue = false;
      }
    }
  }
};
</script>

<style scoped>
.github-integration {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
}

.card {
  margin-top: 2rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  border: 1px solid #e1e4e8;
  border-radius: 6px;
}

.card-body {
  padding: 2rem;
}

.card-title {
  margin-bottom: 1.5rem;
  font-size: 1.25rem;
  font-weight: 500;
}

.btn {
  display: inline-block;
  font-weight: 500;
  text-align: center;
  white-space: nowrap;
  vertical-align: middle;
  user-select: none;
  border: 1px solid transparent;
  padding: 0.5rem 1rem;
  font-size: 1rem;
  line-height: 1.5;
  border-radius: 0.25rem;
  transition: all 0.15s ease-in-out;
  cursor: pointer;
}

.btn-primary {
  color: #fff;
  background-color: #2ea44f;
  border-color: #2ea44f;
}

.btn-primary:hover {
  background-color: #2c974b;
  border-color: #2c974b;
}

.btn-primary:disabled {
  background-color: #94d3a2;
  border-color: #94d3a2;
  cursor: not-allowed;
}

.alert {
  padding: 1rem;
  margin-bottom: 1rem;
  border: 1px solid transparent;
  border-radius: 6px;
}

.alert-success {
  color: #155724;
  background-color: #d4edda;
  border-color: #c3e6cb;
}

.alert h4 {
  margin-top: 0;
  margin-bottom: 0.5rem;
  font-size: 1.1rem;
}

.alert p {
  margin-bottom: 0;
}

a {
  color: #0366d6;
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

.mt-3 {
  margin-top: 1rem !important;
}

.mt-4 {
  margin-top: 1.5rem !important;
}
</style>
