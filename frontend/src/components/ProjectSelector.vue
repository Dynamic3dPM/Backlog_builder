<template>
  <div class="project-selector">
    <div class="form-group">
      <label for="project-select">Select Project</label>
      <select 
        id="project-select" 
        v-model="selectedProject" 
        @change="onProjectChange" 
        class="form-control"
        :disabled="loading"
      >
        <option value="" disabled>Select a project</option>
        <option 
          v-for="project in projects" 
          :key="project.id" 
          :value="project"
        >
          {{ project.title }}
        </option>
      </select>
    </div>

    <div class="form-group" v-if="selectedProject">
      <label for="repository-select">Select Repository</label>
      <select 
        id="repository-select" 
        v-model="selectedRepository" 
        class="form-control"
        :disabled="loadingRepositories || !selectedProject"
      >
        <option value="" disabled>Select a repository</option>
        <option 
          v-for="repo in repositories" 
          :key="repo.id" 
          :value="repo"
        >
          {{ repo.name }}
        </option>
      </select>
      <small v-if="loadingRepositories" class="text-muted">Loading repositories...</small>
    </div>
  </div>
</template>

<script>
import { toast } from 'vue3-toastify';
import 'vue3-toastify/dist/index.css';
import githubService from '@/services/github';

export default {
  name: 'ProjectSelector',
  setup() {
    return { toast }
  },
  props: {
    value: {
      type: Object,
      default: () => ({
        project: null,
        repository: null
      })
    }
  },
  data() {
    return {
      loading: false,
      loadingRepositories: false,
      projects: [],
      repositories: [],
      selectedProject: null,
      selectedRepository: null
    };
  },
  async created() {
    await this.fetchProjects();
    
    // Initialize with prop values if they exist
    if (this.value.project && this.value.repository) {
      this.selectedProject = this.value.project;
      this.selectedRepository = this.value.repository;
      this.repositories = [this.value.repository];
    }
  },
  methods: {
    async fetchProjects() {
      this.loading = true;
      try {
        this.projects = await githubService.getProjects();
      } catch (error) {
        console.error('Failed to fetch projects:', error);
        this.toast.error('Failed to load projects', { position: 'top-right' });
        this.$emit('error', 'Failed to load projects');
      } finally {
        this.loading = false;
      }
    },
    
    async onProjectChange() {
      if (!this.selectedProject) {
        this.repositories = [];
        this.selectedRepository = null;
        this.emitChange();
        return;
      }

      this.loadingRepositories = true;
      try {
        this.repositories = await githubService.getProjectRepositories(this.selectedProject.id);
        this.selectedRepository = this.repositories.length > 0 ? this.repositories[0] : null;
        this.emitChange();
      } catch (error) {
        console.error('Failed to fetch repositories:', error);
        this.toast.error('Failed to load repositories', { position: 'top-right' });
        this.$emit('error', 'Failed to load repositories');
        this.repositories = [];
        this.selectedRepository = null;
      } finally {
        this.loadingRepositories = false;
      }
    },
    
    emitChange() {
      this.$emit('input', {
        project: this.selectedProject,
        repository: this.selectedRepository
      });
      this.$emit('change', {
        project: this.selectedProject,
        repository: this.selectedRepository
      });
    }
  },
  watch: {
    selectedRepository() {
      this.emitChange();
    },
    value: {
      handler(newVal) {
        if (newVal.project && newVal.repository) {
          this.selectedProject = newVal.project;
          this.selectedRepository = newVal.repository;
        }
      },
      deep: true
    }
  }
};
</script>

<style scoped>
.project-selector {
  margin-bottom: 1.5rem;
}

.form-group {
  margin-bottom: 1rem;
}

label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
}

select {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 1rem;
}

.text-muted {
  color: #6c757d;
  font-size: 0.875rem;
  margin-top: 0.25rem;
  display: block;
}
</style>