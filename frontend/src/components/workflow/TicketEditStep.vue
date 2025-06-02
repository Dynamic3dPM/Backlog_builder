<template>
  <div class="ticket-edit-step">
    <h3 class="mb-4">Edit Tickets</h3>
    <p class="text-muted mb-4">Review and refine your tickets before uploading</p>
    
    <div class="alert alert-info" v-if="tickets.length === 0">
      <i class="bi bi-info-circle me-2"></i>
      No tickets to edit. Please go back and create some tickets first.
    </div>
    
    <div v-else>
      <div class="row mb-4">
        <div class="col-md-4">
          <div class="card h-100">
            <div class="card-header d-flex justify-content-between align-items-center">
              <span>Tickets List</span>
              <span class="badge bg-primary rounded-pill">{{ tickets.length }}</span>
            </div>
            <div class="list-group list-group-flush ticket-list">
              <button
                v-for="(ticket, index) in tickets"
                :key="index"
                class="list-group-item list-group-item-action d-flex justify-content-between align-items-center"
                :class="{ 'active': currentTicketIndex === index }"
                @click="selectTicket(index)"
              >
                <div class="d-flex align-items-center">
                  <span class="badge me-2" :class="getPriorityBadgeClass(ticket.priority)">
                    {{ formatPriority(ticket.priority) }}
                  </span>
                  <span class="ticket-title">{{ ticket.title || 'Untitled Ticket' }}</span>
                </div>
                <i class="bi bi-chevron-right"></i>
              </button>
            </div>
          </div>
        </div>
        
        <div class="col-md-8">
          <div class="card h-100" v-if="currentTicket !== null">
            <div class="card-header d-flex justify-content-between align-items-center">
              <h5 class="mb-0">Edit Ticket #{{ currentTicketIndex + 1 }}</h5>
              <div>
                <button 
                  class="btn btn-sm btn-outline-primary me-2"
                  @click="magicRewrite"
                  :disabled="isRewriting"
                >
                  <i class="bi" :class="isRewriting ? 'bi-arrow-repeat spin' : 'bi-magic'"></i>
                  {{ isRewriting ? 'Rewriting...' : 'Magic Rewrite' }}
                </button>
                <button 
                  class="btn btn-sm btn-outline-danger"
                  @click="deleteCurrentTicket"
                >
                  <i class="bi bi-trash"></i>
                </button>
              </div>
            </div>
            <div class="card-body">
              <form @submit.prevent="saveTicket">
                <div class="mb-3">
                  <label class="form-label">Title</label>
                  <input 
                    type="text" 
                    class="form-control" 
                    v-model="currentTicket.title"
                    placeholder="Enter ticket title"
                    required
                  >
                </div>
                <div class="mb-3">
                  <label class="form-label">Description</label>
                  <textarea 
                    class="form-control" 
                    v-model="currentTicket.description" 
                    rows="5"
                    placeholder="Enter ticket description"
                    required
                  ></textarea>
                </div>
                <div class="row">
                  <div class="col-md-6">
                    <div class="mb-3">
                      <label class="form-label">Type</label>
                      <select class="form-select" v-model="currentTicket.type">
                        <option value="task">Task</option>
                        <option value="bug">Bug</option>
                        <option value="feature">Feature</option>
                        <option value="enhancement">Enhancement</option>
                        <option value="epic">Epic</option>
                        <option value="story">Story</option>
                      </select>
                    </div>
                  </div>
                  <div class="col-md-6">
                    <div class="mb-3">
                      <label class="form-label">Priority</label>
                      <select class="form-select" v-model="currentTicket.priority">
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                        <option value="critical">Critical</option>
                      </select>
                    </div>
                  </div>
                </div>
                <div class="row">
                  <div class="col-md-6">
                    <div class="mb-3">
                      <label class="form-label">Estimated Time (hours)</label>
                      <input 
                        type="number" 
                        class="form-control" 
                        v-model.number="currentTicket.estimatedTime"
                        min="0"
                        step="0.5"
                        required
                      >
                    </div>
                  </div>
                  <div class="col-md-6">
                    <div class="mb-3">
                      <label class="form-label">Status</label>
                      <select class="form-select" v-model="currentTicket.status">
                        <option value="todo">To Do</option>
                        <option value="in-progress">In Progress</option>
                        <option value="in-review">In Review</option>
                        <option value="done">Done</option>
                      </select>
                    </div>
                  </div>
                </div>
                <div class="form-check mb-3">
                  <input 
                    class="form-check-input" 
                    type="checkbox" 
                    v-model="currentTicket.requiresReview"
                    id="requiresReviewEdit"
                  >
                  <label class="form-check-label" for="requiresReviewEdit">
                    Requires code review
                  </label>
                </div>
                <div class="form-check mb-3">
                  <input 
                    class="form-check-input" 
                    type="checkbox" 
                    v-model="currentTicket.includeInUpload"
                    id="includeInUpload"
                  >
                  <label class="form-check-label" for="includeInUpload">
                    Include in upload
                  </label>
                </div>
                <div class="d-flex justify-content-between align-items-center">
                  <div>
                    <button 
                      type="button" 
                      class="btn btn-outline-secondary me-2"
                      @click="prevTicket"
                      :disabled="currentTicketIndex === 0"
                    >
                      <i class="bi bi-chevron-left"></i> Previous
                    </button>
                    <button 
                      type="button" 
                      class="btn btn-outline-secondary"
                      @click="nextTicket"
                      :disabled="currentTicketIndex === tickets.length - 1"
                    >
                      Next <i class="bi bi-chevron-right"></i>
                    </button>
                  </div>
                  <button type="submit" class="btn btn-primary">
                    <i class="bi bi-check-lg me-1"></i> Save Changes
                  </button>
                </div>
              </form>
            </div>
          </div>
          <div v-else class="card h-100 d-flex align-items-center justify-content-center">
            <div class="text-center p-5">
              <i class="bi bi-inbox text-muted" style="font-size: 3rem;"></i>
              <p class="mt-3">Select a ticket to edit</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="d-flex justify-content-between mt-4">
      <button class="btn btn-outline-secondary" @click="$emit('prev')">
        <i class="bi bi-arrow-left me-2"></i> Back to Tickets
      </button>
      <button 
        class="btn btn-primary" 
        @click="handleNext"
        :disabled="tickets.length === 0"
      >
        Next: Upload to Project <i class="bi bi-arrow-right ms-2"></i>
      </button>
    </div>
  </div>
</template>

<script>
import { toast } from 'vue3-toastify';

export default {
  name: 'TicketEditStep',
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
      tickets: this.data.editedTickets?.length ? this.data.editedTickets : (this.data.tickets || []),
      currentTicketIndex: 0,
      isRewriting: false
    };
  },
  computed: {
    currentTicket() {
      return this.tickets[this.currentTicketIndex] || null;
    }
  },
  watch: {
    tickets: {
      handler(newTickets) {
        this.$emit('update:data', { editedTickets: newTickets });
      },
      deep: true
    }
  },
  methods: {
    selectTicket(index) {
      this.currentTicketIndex = index;
    },
    nextTicket() {
      if (this.currentTicketIndex < this.tickets.length - 1) {
        this.currentTicketIndex++;
      }
    },
    prevTicket() {
      if (this.currentTicketIndex > 0) {
        this.currentTicketIndex--;
      }
    },
    saveTicket() {
      // The ticket is already being updated in the array since it's a reference
      toast.success('Ticket updated');
    },
    deleteCurrentTicket() {
      if (confirm('Are you sure you want to delete this ticket?')) {
        this.tickets.splice(this.currentTicketIndex, 1);
        if (this.currentTicketIndex >= this.tickets.length) {
          this.currentTicketIndex = Math.max(0, this.tickets.length - 1);
        }
        toast.info('Ticket deleted');
      }
    },
    async magicRewrite() {
      if (!this.currentTicket) return;
      
      this.isRewriting = true;
      try {
        // Simulate API call to rewrite the ticket
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // In a real app, you would call your backend API here
        // For now, we'll just enhance the current description
        const enhancedDescription = `[Enhanced] ${this.currentTicket.description}\n\n` +
          `Acceptance Criteria:\n` +
          `- [ ] ${this.currentTicket.title} should work as expected\n` +
          `- [ ] All edge cases should be handled\n` +
          `- [ ] Code should be well-documented\n`;
        
        this.currentTicket.description = enhancedDescription;
        toast.success('Ticket enhanced with AI');
      } catch (error) {
        console.error('Error rewriting ticket:', error);
        toast.error('Failed to rewrite ticket');
      } finally {
        this.isRewriting = false;
      }
    },
    getPriorityBadgeClass(priority) {
      const classes = {
        low: 'bg-success',
        medium: 'bg-primary',
        high: 'bg-warning',
        critical: 'bg-danger'
      };
      return classes[priority] || 'bg-secondary';
    },
    formatPriority(priority) {
      return priority.charAt(0).toUpperCase() + priority.slice(1);
    },
    normalizePriority(priority) {
      if (!priority) return 'medium';
      const normalized = priority.toLowerCase();
      const validPriorities = ['low', 'medium', 'high', 'critical'];
      return validPriorities.includes(normalized) ? normalized : 'medium';
    },
    normalizeType(type) {
      if (!type) return 'task';
      const normalized = type.toLowerCase();
      const typeMapping = {
        'task': 'task',
        'bug': 'bug', 
        'feature': 'feature',
        'enhancement': 'enhancement',
        'epic': 'epic',
        'story': 'story'
      };
      return typeMapping[normalized] || 'task';
    },
    cleanMarkdown(text) {
      if (!text) return '';
      return text
        .replace(/\*\*(.*?)\*\*/g, '$1')  // Remove **bold**
        .replace(/\*(.*?)\*/g, '$1')      // Remove *italic*
        .replace(/###\s+/g, '')           // Remove ### headers with space
        .replace(/##\s+/g, '')            // Remove ## headers with space
        .replace(/#\s+/g, '')             // Remove # headers with space
        .replace(/`(.*?)`/g, '$1')        // Remove `code`
        .replace(/^\s*##\s*/gm, '')       // Remove ## at start of lines
        .replace(/^\s*\*\*.*?\*\*\s*$/gm, '') // Remove lines that are just **text**
        .replace(/\*\*([^*]+)\*\*/g, '$1') // Remove any remaining **bold**
        .split('\n')
        .filter(line => {
          const trimmed = line.trim();
          // Filter out lines that are just markdown metadata
          return !trimmed.startsWith('**') && 
                 !trimmed.startsWith('##') && 
                 !trimmed.startsWith('*') &&
                 trimmed !== '';
        })
        .join('\n')
        .trim();
    },
    handleNext() {
      if (this.tickets.length === 0) {
        toast.warning('Please create at least one ticket');
        return;
      }
      this.$emit('next');
    }
  },
  mounted() {
    // Initialize includeInUpload flag for all tickets if not set
    this.tickets = this.tickets.map(ticket => ({
      ...ticket,
      includeInUpload: ticket.includeInUpload !== false,
      status: ticket.status || 'todo',
      priority: this.normalizePriority(ticket.priority),
      type: this.normalizeType(ticket.type),
      // Clean markdown from title and description
      title: this.cleanMarkdown(ticket.title),
      description: this.cleanMarkdown(ticket.description)
    }));
  }
};
</script>

<style scoped>
.ticket-list {
  max-height: 600px;
  overflow-y: auto;
}

.ticket-title {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 200px;
  display: inline-block;
}

.list-group-item {
  border-left: none;
  border-right: none;
  border-radius: 0 !important;
  cursor: pointer;
  transition: all 0.2s ease;
}

.list-group-item:first-child {
  border-top: none;
}

.list-group-item.active {
  background-color: #f8f9fa;
  color: #212529;
  border-color: #dee2e6;
  border-left: 3px solid #0d6efd !important;
}

.list-group-item:hover:not(.active) {
  background-color: #f8f9fa;
}

.spin {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Custom scrollbar */
.ticket-list::-webkit-scrollbar {
  width: 6px;
}

.ticket-list::-webkit-scrollbar-track {
  background: #f1f1f1;
}

.ticket-list::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 3px;
}

.ticket-list::-webkit-scrollbar-thumb:hover {
  background: #555;
}

/* Make sure the badge text is white */
.badge {
  color: white !important;
}
</style>
