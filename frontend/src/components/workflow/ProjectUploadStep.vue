<template>
  <div class="project-upload-step">
    <h3 class="mb-4">Upload to Project</h3>
    <p class="text-muted mb-4">Select a GitHub repository and project to upload your tickets</p>
    
    <div v-if="tickets.length === 0" class="alert alert-warning">
      <i class="bi bi-exclamation-triangle me-2"></i>
      No tickets to upload. Please go back and create some tickets first.
    </div>
    
    <div v-else>
      <div class="card mb-4">
        <div class="card-header">
          <h5 class="mb-0">Selected Tickets</h5>
        </div>
        <div class="card-body">
          <div class="table-responsive">
            <table class="table table-hover">
              <thead>
                <tr>
                  <th width="40">#</th>
                  <th>Title</th>
                  <th>Type</th>
                  <th>Priority</th>
                  <th>Est. Time</th>
                  <th>Status</th>
                  <th>Include</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(ticket, index) in tickets" :key="index">
                  <td>{{ index + 1 }}</td>
                  <td>
                    <div class="d-flex align-items-center">
                      <i class="bi" :class="getTicketIcon(ticket.type)" :title="ticket.type"></i>
                      <span class="ms-2">{{ ticket.title || 'Untitled Ticket' }}</span>
                    </div>
                  </td>
                  <td>
                    <span class="badge" :class="getTypeBadgeClass(ticket.type)">
                      {{ formatType(ticket.type) }}
                    </span>
                  </td>
                  <td>
                    <span class="badge" :class="getPriorityBadgeClass(ticket.priority)">
                      {{ formatPriority(ticket.priority) }}
                    </span>
                  </td>
                  <td>{{ ticket.estimatedTime || 'N/A' }}h</td>
                  <td>
                    <span class="badge" :class="getStatusBadgeClass(ticket.status)">
                      {{ formatStatus(ticket.status) }}
                    </span>
                  </td>
                  <td class="text-center">
                    <div class="form-check d-inline-block">
                      <input 
                        class="form-check-input" 
                        type="checkbox" 
                        v-model="ticket.includeInUpload"
                        :id="'include-' + index"
                      >
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div class="d-flex justify-content-between align-items-center mt-3">
            <div>
              <span class="me-3">{{ selectedTicketsCount }} of {{ tickets.length }} selected</span>
            </div>
            <button 
              class="btn btn-outline-secondary btn-sm"
              @click="toggleSelectAll"
            >
              {{ allSelected ? 'Deselect All' : 'Select All' }}
            </button>
          </div>
        </div>
      </div>
      
      <div class="row">
        <div class="col-md-6">
          <div class="card mb-4">
            <div class="card-header">
              <h5 class="mb-0">GitHub Repository</h5>
            </div>
            <div class="card-body">
              <div v-if="isLoadingRepos" class="text-center py-3">
                <div class="spinner-border text-primary" role="status">
                  <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2 mb-0">Loading repositories...</p>
              </div>
              <div v-else-if="reposError" class="alert alert-danger">
                <i class="bi bi-exclamation-triangle me-2"></i>
                {{ reposError }}
              </div>
              <div v-else>
                <div class="mb-3">
                  <label class="form-label">Select Repository</label>
                  <select 
                    class="form-select" 
                    v-model="selectedRepo"
                    :disabled="isLoadingRepos"
                  >
                    <option value="" disabled>Select a repository</option>
                    <option 
                      v-for="repo in repos" 
                      :key="repo.id" 
                      :value="repo"
                      :disabled="!repo.has_issues"
                    >
                      {{ repo.full_name }}
                      <template v-if="!repo.has_issues"> (Issues disabled)</template>
                    </option>
                  </select>
                  <div v-if="selectedRepo && !selectedRepo.has_issues" class="text-danger small mt-1">
                    <i class="bi bi-exclamation-circle me-1"></i>
                    Issues are disabled for this repository. Please select another one or enable issues.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div class="col-md-6">
          <div class="card mb-4">
            <div class="card-header">
              <h5 class="mb-0">GitHub Project</h5>
            </div>
            <div class="card-body">
              <div v-if="isLoadingProjects" class="text-center py-3">
                <div class="spinner-border text-primary" role="status">
                  <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2 mb-0">Loading projects...</p>
              </div>
              <div v-else-if="projectsError" class="alert alert-danger">
                <i class="bi bi-exclamation-triangle me-2"></i>
                {{ projectsError }}
              </div>
              <div v-else>
                <div class="mb-3">
                  <label class="form-label">Select Project</label>
                  <select 
                    class="form-select" 
                    v-model="selectedProject"
                    :disabled="isLoadingProjects || !selectedRepo"
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
                
                <div v-if="selectedProject" class="mt-3">
                  <h6>Project Details:</h6>
                  <ul class="list-unstyled small">
                    <li><strong>Description:</strong> {{ selectedProject.description || 'No description' }}</li>
                    <li><strong>Created:</strong> {{ formatDate(selectedProject.createdAt) }}</li>
                    <li><strong>Updated:</strong> {{ formatDate(selectedProject.updatedAt) }}</li>
                    <li><strong>State:</strong> {{ selectedProject.closed ? 'Closed' : 'Open' }}</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div class="card mb-4">
        <div class="card-header">
          <h5 class="mb-0">Upload Options</h5>
        </div>
        <div class="card-body">
          <div class="row">
            <div class="col-md-6">
              <div class="mb-3">
                <label class="form-label">Default Assignee</label>
                <input 
                  type="text" 
                  class="form-control" 
                  v-model="uploadOptions.assignee"
                  placeholder="GitHub username (optional)"
                >
              </div>
            </div>
            <div class="col-md-6">
              <div class="mb-3">
                <label class="form-label">Labels</label>
                <input 
                  type="text" 
                  class="form-control" 
                  v-model="uploadOptions.labels"
                  placeholder="comma,separated,labels"
                >
              </div>
            </div>
          </div>
          <div class="mb-3">
            <div class="form-check">
              <input 
                class="form-check-input" 
                type="checkbox" 
                v-model="uploadOptions.createAsDraft"
                id="createAsDraft"
              >
              <label class="form-check-label" for="createAsDraft">
                Create as draft (if supported)
              </label>
            </div>
          </div>
          <div class="alert alert-info">
            <i class="bi bi-info-circle me-2"></i>
            <strong>Note:</strong> Only tickets marked as "Include" will be uploaded to the selected project.
          </div>
        </div>
      </div>
      
      <div class="d-flex justify-content-between">
        <button class="btn btn-outline-secondary" @click="$emit('prev')">
          <i class="bi bi-arrow-left me-2"></i> Back to Editing
        </button>
        <div>
          <button 
            class="btn btn-outline-secondary me-2"
            @click="previewIssues"
            :disabled="!canPreview"
          >
            <i class="bi bi-eye me-1"></i> Preview
          </button>
          <button 
            class="btn btn-primary" 
            @click="uploadToGitHub"
            :disabled="!canUpload || isUploading"
          >
            <span v-if="isUploading">
              <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
              Uploading...
            </span>
            <span v-else>
              <i class="bi bi-github me-1"></i> Upload to GitHub
            </span>
          </button>
        </div>
      </div>
    </div>
    
    <!-- Preview Modal -->
    <div class="modal fade" :class="{ 'show d-block': showPreviewModal }" tabindex="-1" v-if="showPreviewModal">
      <div class="modal-dialog modal-xl">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title">Preview GitHub Issues</h5>
            <button type="button" class="btn-close" @click="showPreviewModal = false"></button>
          </div>
          <div class="modal-body">
            <div class="alert alert-info mb-4">
              <i class="bi bi-info-circle me-2"></i>
              This is a preview of how the issues will appear in GitHub.
            </div>
            
            <div v-for="(ticket, index) in ticketsToUpload" :key="index" class="card mb-3">
              <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="mb-0">
                  <i class="bi" :class="getTicketIcon(ticket.type)" :title="ticket.type"></i>
                  {{ ticket.title || 'Untitled Ticket' }}
                </h5>
                <div>
                  <span class="badge me-2" :class="getTypeBadgeClass(ticket.type)">
                    {{ formatType(ticket.type) }}
                  </span>
                  <span class="badge" :class="getPriorityBadgeClass(ticket.priority)">
                    {{ formatPriority(ticket.priority) }}
                  </span>
                </div>
              </div>
              <div class="card-body">
                <h6>Description:</h6>
                <div class="bg-light p-3 mb-3 rounded" style="white-space: pre-line">
                  {{ ticket.description || 'No description provided.' }}
                </div>
                <div class="row">
                  <div class="col-md-6">
                    <p class="mb-1"><strong>Estimated Time:</strong> {{ ticket.estimatedTime || 'N/A' }} hours</p>
                    <p class="mb-1"><strong>Requires Review:</strong> {{ ticket.requiresReview ? 'Yes' : 'No' }}</p>
                  </div>
                  <div class="col-md-6">
                    <p class="mb-1"><strong>Labels:</strong> {{ uploadOptions.labels || 'None' }}</p>
                    <p class="mb-0"><strong>Assignee:</strong> {{ uploadOptions.assignee || 'None' }}</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div v-if="ticketsToUpload.length === 0" class="text-center py-5 bg-light rounded">
              <i class="bi bi-inbox text-muted" style="font-size: 3rem;"></i>
              <p class="mt-3">No tickets selected for upload</p>
            </div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" @click="showPreviewModal = false">
              Close
            </button>
            <button 
              type="button" 
              class="btn btn-primary" 
              @click="confirmUpload"
              :disabled="ticketsToUpload.length === 0 || isUploading"
            >
              <span v-if="isUploading">
                <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                Uploading...
              </span>
              <span v-else>
                <i class="bi bi-check-lg me-1"></i> Confirm Upload
              </span>
            </button>
          </div>
        </div>
      </div>
    </div>
    <div class="modal-backdrop fade show" v-if="showPreviewModal"></div>
  </div>
</template>

<script>
export default {
  name: 'ProjectUploadStep',
  props: {
    data: {
      type: Object,
      required: true
    },
    currentStep: {
      type: Number,
      required: true
    }
  },
  emits: ['next', 'prev', 'update:data'],
  data() {
    return {
      tickets: [...this.data.editedTickets || []],
      repos: [],
      projects: [],
      selectedRepo: null,
      selectedProject: null,
      isLoadingRepos: false,
      isLoadingProjects: false,
      reposError: null,
      projectsError: null,
      isUploading: false,
      showPreviewModal: false,
      uploadOptions: {
        assignee: '',
        labels: 'backlog',
        createAsDraft: true
      }
    };
  },
  computed: {
    selectedTicketsCount() {
      return this.tickets.filter(t => t.includeInUpload !== false).length;
    },
    allSelected: {
      get() {
        return this.tickets.length > 0 && this.tickets.every(t => t.includeInUpload !== false);
      },
      set(value) {
        this.tickets = this.tickets.map(ticket => ({
          ...ticket,
          includeInUpload: value
        }));
      }
    },
    ticketsToUpload() {
      return this.tickets.filter(t => t.includeInUpload !== false);
    },
    canPreview() {
      return this.selectedRepo && this.selectedProject && this.ticketsToUpload.length > 0;
    },
    canUpload() {
      return this.selectedRepo && 
             this.selectedRepo.has_issues && 
             this.selectedProject && 
             this.ticketsToUpload.length > 0;
    }
  },
  watch: {
    selectedRepo(newRepo) {
      this.fetchProjects(newRepo);
    },
    tickets: {
      handler(newTickets) {
        this.$emit('update:data', { editedTickets: newTickets });
      },
      deep: true
    }
  },
  methods: {
    toggleSelectAll() {
      this.allSelected = !this.allSelected;
    },
    getTicketIcon(type) {
      const icons = {
        task: 'bi-check-square',
        bug: 'bi-bug',
        feature: 'bi-star',
        enhancement: 'bi-lightbulb'
      };
      return icons[type] || 'bi-card-text';
    },
    getTypeBadgeClass(type) {
      const classes = {
        task: 'bg-primary',
        bug: 'bg-danger',
        feature: 'bg-success',
        enhancement: 'bg-info'
      };
      return `${classes[type] || 'bg-secondary'} text-white`;
    },
    getPriorityBadgeClass(priority) {
      const classes = {
        low: 'bg-success',
        medium: 'bg-primary',
        high: 'bg-warning',
        critical: 'bg-danger'
      };
      return `${classes[priority] || 'bg-secondary'} text-white`;
    },
    getStatusBadgeClass(status) {
      const classes = {
        'todo': 'bg-secondary',
        'in-progress': 'bg-primary',
        'in-review': 'bg-info',
        'done': 'bg-success'
      };
      return `${classes[status] || 'bg-light text-dark'}`;
    },
    formatType(type) {
      return type ? type.charAt(0).toUpperCase() + type.slice(1) : '';
    },
    formatPriority(priority) {
      return priority ? priority.charAt(0).toUpperCase() + priority.slice(1) : '';
    },
    formatStatus(status) {
      if (!status) return '';
      return status
        .split('-')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
    },
    formatDate(dateString) {
      if (!dateString) return 'N/A';
      return new Date(dateString).toLocaleDateString();
    },
    async fetchRepositories() {
      this.isLoadingRepos = true;
      this.reposError = null;
      
      try {
        const response = await fetch('/api/github/repositories');
        if (!response.ok) {
          throw new Error('Failed to fetch repositories');
        }
        
        const data = await response.json();
        this.repos = data.repositories || [];
        
        // Auto-select the first repo with issues enabled if available
        const firstRepoWithIssues = this.repos.find(repo => repo.has_issues);
        if (firstRepoWithIssues) {
          this.selectedRepo = firstRepoWithIssues;
        }
      } catch (error) {
        console.error('Error fetching repositories:', error);
        this.reposError = 'Failed to load repositories. Please try again later.';
        
        // Fallback to mock data for demo purposes
        this.repos = [
          {
            id: 1,
            name: 'ffpinvest-mobile-app',
            full_name: 'Dynamic3D/ffpinvest-mobile-app',
            has_issues: true,
            private: true
          },
          {
            id: 2,
            name: 'SaddleFitBoard',
            full_name: 'Dynamic3D/SaddleFitBoard',
            has_issues: false,
            private: true
          },
          {
            id: 3,
            name: 'Saddle-fit-intro-website',
            full_name: 'Dynamic3D/Saddle-fit-intro-website',
            has_issues: true,
            private: false
          }
        ];
        
        const firstRepoWithIssues = this.repos.find(repo => repo.has_issues);
        if (firstRepoWithIssues) {
          this.selectedRepo = firstRepoWithIssues;
        }
      } finally {
        this.isLoadingRepos = false;
      }
    },
    async fetchProjects(repo) {
      if (!repo) {
        this.projects = [];
        this.selectedProject = null;
        return;
      }
      
      this.isLoadingProjects = true;
      this.projectsError = null;
      
      try {
        const response = await fetch('/api/github/projects');
        if (!response.ok) {
          throw new Error('Failed to fetch projects');
        }
        
        const data = await response.json();
        this.projects = data.projects || [];
        
        // Auto-select the first project if available
        if (this.projects.length > 0) {
          this.selectedProject = this.projects[0];
        }
      } catch (error) {
        console.error('Error fetching projects:', error);
        this.projectsError = 'Failed to load projects. Please try again later.';
        
        // Fallback to mock data for demo purposes
        this.projects = [
          {
            id: 'PVT_kwDOCKI6dM4A0RR-',
            number: 1,
            name: 'SaddleFit',
            body: 'Main project board for SaddleFit application',
            state: 'open',
            created_at: '2023-01-15T10:00:00Z',
            updated_at: '2023-06-01T14:30:00Z'
          },
          {
            id: 'PVT_kwDOCKI6dM4A0RR1',
            number: 2,
            name: 'Sprint 1',
            body: 'Sprint 1 Backlog',
            state: 'open',
            created_at: '2023-02-01T09:15:00Z',
            updated_at: '2023-05-15T11:20:00Z'
          }
        ];
        
        if (this.projects.length > 0) {
          this.selectedProject = this.projects[0];
        }
      } finally {
        this.isLoadingProjects = false;
      }
    },
    previewIssues() {
      if (this.canPreview) {
        this.showPreviewModal = true;
      }
    },
    uploadToGitHub() {
      if (!this.canUpload) return;
      this.showPreviewModal = true;
    },
    async confirmUpload() {
      alert('NEW CODE: confirmUpload called - if you see this, the new code is loaded');
      console.log('confirmUpload called');
      console.log('ticketsToUpload:', this.ticketsToUpload);
      console.log('selectedRepo:', this.selectedRepo);
      console.log('selectedProject:', this.selectedProject);
      
      if (this.ticketsToUpload.length === 0 || !this.selectedRepo || !this.selectedProject) {
        this.$toast.warning('Please select at least one ticket and ensure both repository and project are selected.');
        return;
      }
      
      this.isUploading = true;
      
      try {
        let issuesData;
        try {
          console.log('Starting ticket mapping...');
          issuesData = this.ticketsToUpload.map(ticket => {
            console.log('Processing ticket:', ticket);
            return {
              title: ticket.title,
              body: ticket.description,
              labels: this.uploadOptions.labels ? this.uploadOptions.labels.split(',').map(l => l.trim()) : [],
              assignees: this.uploadOptions.assignee ? [this.uploadOptions.assignee] : []
            };
          });
          console.log('Mapped issues data:', issuesData);
        } catch (error) {
          console.error('Error mapping tickets:', error);
          this.$toast.error('Failed to map tickets for upload. Please try again later.');
          this.isUploading = false;
          return;
        }
        
        const requestPayload = {
          repository: this.selectedRepo.name,
          projectId: this.selectedProject.id,
          tickets: issuesData
        };
        
        console.log('Upload request payload:', requestPayload);
        console.log('Selected repo:', this.selectedRepo);
        console.log('Selected project:', this.selectedProject);
        
        const response = await fetch('/api/github/create-issues', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(requestPayload)
        });
        
        console.log('Response status:', response.status);
        console.log('Response ok:', response.ok);
        
        if (!response.ok) {
          const errorText = await response.text();
          console.error('HTTP Error Response:', errorText);
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        
        let result;
        try {
          const responseText = await response.text();
          console.log('Raw response text:', responseText);
          
          if (!responseText) {
            throw new Error('Empty response from server');
          }
          
          result = JSON.parse(responseText);
          console.log('Parsed result:', result);
        } catch (parseError) {
          console.error('JSON parsing error:', parseError);
          throw new Error(`Failed to parse server response: ${parseError.message}`);
        }
        
        // Check if result has the expected structure
        if (!result || !result.summary) {
          throw new Error('Invalid response format from server');
        }
        
        this.$toast.success(`Successfully created ${result.summary.successful} of ${result.summary.total} ${result.summary.total === 1 ? 'issue' : 'issues'} in ${this.selectedRepo.full_name}`);
        
        // Update workflow data with upload results
        this.$emit('update:data', { 
          selectedProject: this.selectedProject,
          selectedRepo: this.selectedRepo,
          uploadResults: result
        });
        
        this.showPreviewModal = false;
        
        // Optionally navigate to next step or show completion
        this.$emit('next');
        
      } catch (error) {
        console.error('Error uploading to GitHub:', error);
        console.error('Error details:', {
          message: error.message,
          stack: error.stack,
          name: error.name
        });
        this.$toast.error(`Failed to upload tickets to GitHub: ${error.message || 'Unknown error'}`);
      } finally {
        this.isUploading = false;
      }
    }
  },
  mounted() {
    console.log('ProjectUploadStep mounted - version with enhanced logging');
    // Initialize includeInUpload flag for all tickets if not set
    this.tickets = this.tickets.map(ticket => ({
      ...ticket,
      includeInUpload: ticket.includeInUpload !== false,
      status: ticket.status || 'todo'
    }));
    
    // Load repositories when component mounts
    this.fetchRepositories();
  }
};
</script>

<style scoped>
.table-responsive {
  max-height: 400px;
  overflow-y: auto;
}

.badge {
  font-size: 0.75rem;
}

.modal {
  z-index: 1050;
  background-color: rgba(0, 0, 0, 0.5);
}

.modal-backdrop {
  z-index: 1040;
}

.spinner-border-sm {
  width: 1rem;
  height: 1rem;
}

.card {
  border: 1px solid rgba(0, 0, 0, 0.125);
  border-radius: 0.5rem;
  box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
}

.card-header {
  background-color: #f8f9fa;
  border-bottom: 1px solid rgba(0, 0, 0, 0.125);
  font-weight: 500;
}

/* Custom scrollbar */
.table-responsive::-webkit-scrollbar {
  width: 8px;
}

.table-responsive::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

.table-responsive::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 4px;
}

.table-responsive::-webkit-scrollbar-thumb:hover {
  background: #555;
}
</style>
