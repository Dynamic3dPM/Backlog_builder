<template>
  <div class="ticket-creation-step">
    <h3 class="mb-4">Create Tickets</h3>
    <p class="text-muted mb-4">Review and adjust the automatically generated tickets</p>
    
    <div class="mb-4">
      <div class="d-flex justify-content-between align-items-center mb-3">
        <h5 class="mb-0">Generated Tickets</h5>
        <button 
          class="btn btn-sm btn-outline-primary"
          @click="regenerateTickets"
          :disabled="isGenerating"
        >
          <i class="bi" :class="isGenerating ? 'bi-arrow-repeat spin' : 'bi-arrow-repeat'"></i>
          Regenerate All
        </button>
      </div>
      
      <div v-if="tickets.length === 0" class="text-center py-5 bg-light rounded">
        <i class="bi bi-inbox text-muted" style="font-size: 3rem;"></i>
        <p class="mt-3">No tickets generated yet</p>
        <button class="btn btn-primary" @click="generateTickets">
          Generate Tickets
        </button>
      </div>
      
      <div v-else>
        <div class="ticket-list">
          <div 
            v-for="(ticket, index) in tickets" 
            :key="index"
            class="ticket-item card mb-3"
            :class="{ 'border-primary': selectedTicketIndex === index }"
            @click="selectTicket(index)"
          >
            <div class="card-body">
              <div class="d-flex justify-content-between">
                <h5 class="card-title mb-1">{{ ticket.title || 'Untitled Ticket' }}</h5>
                <div class="form-check">
                  <input 
                    class="form-check-input" 
                    type="checkbox" 
                    :id="'ticket-' + index"
                    :checked="ticket.selected !== false"
                    @click.stop
                    @change="toggleTicketSelection(index, $event)"
                  >
                  <label class="form-check-label" :for="'ticket-' + index">
                    Include
                  </label>
                </div>
              </div>
              <p class="card-text text-muted small mb-2">
                {{ ticket.description ? ticket.description.substring(0, 120) + '...' : 'No description' }}
              </p>
              <div class="d-flex justify-content-between align-items-center">
                <span class="badge bg-secondary">#{{ index + 1 }}</span>
                <div>
                  <button 
                    class="btn btn-sm btn-outline-secondary me-1"
                    @click.stop="editTicket(index)"
                  >
                    <i class="bi bi-pencil"></i>
                  </button>
                  <button 
                    class="btn btn-sm btn-outline-danger"
                    @click.stop="deleteTicket(index)"
                  >
                    <i class="bi bi-trash"></i>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div class="d-flex justify-content-between mt-3">
          <button 
            class="btn btn-outline-primary"
            @click="addNewTicket"
          >
            <i class="bi bi-plus-lg me-1"></i> Add New Ticket
          </button>
          <div>
            <span class="me-3">{{ selectedTicketsCount }} of {{ tickets.length }} selected</span>
            <button 
              class="btn btn-danger"
              :disabled="selectedTicketsCount === 0"
              @click="deleteSelectedTickets"
            >
              <i class="bi bi-trash me-1"></i> Delete Selected
            </button>
          </div>
        </div>
      </div>
    </div>

    <div class="d-flex justify-content-between mt-4">
      <button class="btn btn-outline-secondary" @click="$emit('prev')">
        <i class="bi bi-arrow-left me-2"></i> Back
      </button>
      <button 
        class="btn btn-primary" 
        @click="handleNext"
        :disabled="tickets.length === 0"
      >
        Next: Edit Tickets <i class="bi bi-arrow-right ms-2"></i>
      </button>
    </div>
    
    <!-- Edit Ticket Modal -->
    <div class="modal" :class="{ 'show d-block': showEditModal }" tabindex="-1" v-if="showEditModal">
      <div class="modal-dialog modal-lg">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title">{{ editingTicketIndex === -1 ? 'Add New' : 'Edit' }} Ticket</h5>
            <button type="button" class="btn-close" @click="closeModal"></button>
          </div>
          <div class="modal-body">
            <div class="mb-3">
              <label class="form-label">Title</label>
              <input 
                type="text" 
                class="form-control" 
                v-model="editingTicket.title"
                placeholder="Enter ticket title"
              >
            </div>
            <div class="mb-3">
              <label class="form-label">Description</label>
              <textarea 
                class="form-control" 
                v-model="editingTicket.description" 
                rows="5"
                placeholder="Enter ticket description"
              ></textarea>
            </div>
            <div class="mb-3">
              <label class="form-label">Type</label>
              <select class="form-select" v-model="editingTicket.type">
                <option value="task">Task</option>
                <option value="bug">Bug</option>
                <option value="feature">Feature</option>
                <option value="enhancement">Enhancement</option>
              </select>
            </div>
            <div class="row">
              <div class="col-md-6">
                <div class="mb-3">
                  <label class="form-label">Priority</label>
                  <select class="form-select" v-model="editingTicket.priority">
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="critical">Critical</option>
                  </select>
                </div>
              </div>
              <div class="col-md-6">
                <div class="mb-3">
                  <label class="form-label">Estimated Time (hours)</label>
                  <input 
                    type="number" 
                    class="form-control" 
                    v-model.number="editingTicket.estimatedTime"
                    min="0"
                    step="0.5"
                  >
                </div>
              </div>
            </div>
            <div class="form-check mb-3">
              <input 
                class="form-check-input" 
                type="checkbox" 
                v-model="editingTicket.requiresReview"
                id="requiresReview"
              >
              <label class="form-check-label" for="requiresReview">
                Requires code review
              </label>
            </div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-outline-secondary" @click="closeModal">
              Cancel
            </button>
            <button 
              type="button" 
              class="btn btn-primary" 
              @click="saveTicket"
              :disabled="!editingTicket.title.trim()"
            >
              Save Ticket
            </button>
          </div>
        </div>
      </div>
    </div>
    <div class="modal-backdrop fade show" v-if="showEditModal"></div>
  </div>
</template>

<script>
export default {
  name: 'TicketCreationStep',
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
      tickets: this.data.tickets || [],
      isGenerating: false,
      selectedTicketIndex: -1,
      showEditModal: false,
      editingTicketIndex: -1,
      editingTicket: this.getEmptyTicket(),
      conversationAnalysis: null
    };
  },
  computed: {
    selectedTicketsCount() {
      return this.tickets.filter(t => t.selected !== false).length;
    }
  },
  watch: {
    tickets: {
      handler(newTickets) {
        this.$emit('update:data', { tickets: newTickets });
      },
      deep: true
    }
  },
  methods: {
    getEmptyTicket() {
      return {
        title: '',
        description: '',
        type: 'task',
        priority: 'medium',
        estimatedTime: 1,
        requiresReview: false,
        selected: true
      };
    },
    async generateTickets() {
      if (!this.data.transcription) {
        this.$toast?.error('No transcription available to generate tickets');
        return;
      }
      
      this.isGenerating = true;
      
      try {
        console.log('Using smart conversation analysis for intelligent ticket generation');
        
        // Use the new smart conversation analysis endpoint that generates contextual tickets
        const smartAnalysisResponse = await fetch('/api/ai/analyze-conversation-smart', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            transcript: this.data.transcription
          })
        });
        
        if (!smartAnalysisResponse.ok) {
          const errorText = await smartAnalysisResponse.text();
          console.error('Smart analysis response error:', errorText);
          throw new Error(`Failed to analyze conversation: ${smartAnalysisResponse.status} ${errorText}`);
        }
        
        const smartResult = await smartAnalysisResponse.json();
        console.log('Smart analysis result:', smartResult);
        
        if (!smartResult.success) {
          throw new Error(smartResult.error || 'Smart analysis failed');
        }
        
        // Extract intelligent tickets from the smart analysis
        const intelligentTickets = smartResult.intelligent_tickets || [];
        
        // Update tickets with the intelligent data
        this.tickets = intelligentTickets.map((ticket, index) => ({
          id: ticket.id || index + 1,
          title: ticket.title || `Ticket ${index + 1}`,
          description: ticket.description || '',
          priority: this.capitalizeFirst(ticket.priority) || 'Medium',
          labels: ticket.labels || [],
          assignee: ticket.assignee || null,
          deadline: ticket.deadline || null,
          dependencies: ticket.dependencies || [],
          acceptance_criteria: ticket.acceptance_criteria || [],
          context_notes: ticket.context_notes || '',
          confidence: ticket.confidence || 0.5,
          selected: true
        }));
        
        // Store conversation analysis for potential use
        this.conversationAnalysis = smartResult.conversation_analysis;
        
        const ticketCount = this.tickets.length;
        const insightCount = smartResult.insights_analyzed || 0;
        
        this.$toast?.success(`Generated ${ticketCount} intelligent tickets from ${insightCount} conversation insights`);
        
        // Log analysis summary for debugging
        console.log('Conversation Analysis Summary:', {
          project_context: this.conversationAnalysis?.project_context,
          team_dynamics: this.conversationAnalysis?.team_dynamics,
          insights_found: this.conversationAnalysis?.insights_found,
          actionable_insights: this.conversationAnalysis?.actionable_insights
        });
        
      } catch (error) {
        console.error('Error generating intelligent tickets:', error);
        this.$toast?.error('Failed to generate intelligent tickets. Using fallback method.');
        
        // Fallback to traditional analysis if smart analysis fails
        await this.generateTicketsFallback();
      } finally {
        this.isGenerating = false;
      }
    },
    
    async generateTicketsFallback() {
      try {
        console.log('Using fallback ticket generation method');
        
        // First, analyze the transcription to extract action items
        const analysisResponse = await fetch('/api/ai/analyze', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            transcription: this.data.transcription,
            meetingType: 'general',
            generateTickets: false
          })
        });
        
        if (!analysisResponse.ok) {
          throw new Error(`Analysis failed: ${analysisResponse.status}`);
        }
        
        const analysisResult = await analysisResponse.json();
        
        // Now generate tickets from the analysis
        const ticketsResponse = await fetch('/api/ai/generate-tickets', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            analysis: analysisResult,
            transcription: this.data.transcription
          })
        });
        
        if (!ticketsResponse.ok) {
          throw new Error(`Ticket generation failed: ${ticketsResponse.status}`);
        }
        
        const ticketResult = await ticketsResponse.json();
        
        // Update tickets with generated data
        this.tickets = (ticketResult.tickets || []).map((ticket, index) => ({
          id: ticket.id || index + 1,
          title: ticket.title || `Ticket ${index + 1}`,
          description: ticket.description || ticket.body || '',
          priority: ticket.priority || 'Medium',
          labels: ticket.labels || [],
          selected: true
        }));
        
        this.$toast?.success(`Generated ${this.tickets.length} tickets using fallback method`);
        
      } catch (fallbackError) {
        console.error('Fallback ticket generation failed:', fallbackError);
        this.$toast?.error('All ticket generation methods failed. Using mock data.');
        
        // Final fallback to mock tickets
        this.tickets = [
          {
            id: 1,
            title: 'Implement User Authentication',
            description: 'Create a secure login system with JWT tokens and password hashing.',
            priority: 'High',
            labels: ['authentication', 'security'],
            selected: true
          },
          {
            id: 2,
            title: 'Design Dashboard UI',
            description: 'Create a responsive dashboard with charts and data visualization.',
            priority: 'Medium',
            labels: ['ui', 'dashboard'],
            selected: true
          },
          {
            id: 3,
            title: 'Setup Database Schema',
            description: 'Design and implement the database schema for user data and analytics.',
            priority: 'High',
            labels: ['database', 'backend'],
            selected: true
          }
        ];
      }
    },
    
    capitalizeFirst(str) {
      if (!str) return str;
      return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
    },
    regenerateTickets() {
      this.tickets = [];
      this.generateTickets();
    },
    addNewTicket() {
      this.editingTicketIndex = -1;
      this.editingTicket = this.getEmptyTicket();
      this.showEditModal = true;
    },
    editTicket(index) {
      this.editingTicketIndex = index;
      this.editingTicket = { ...this.tickets[index] };
      this.showEditModal = true;
    },
    saveTicket() {
      if (this.editingTicketIndex === -1) {
        // Add new ticket
        this.tickets.push({ ...this.editingTicket });
      } else {
        // Update existing ticket
        this.tickets[this.editingTicketIndex] = { ...this.editingTicket };
      }
      this.closeModal();
    },
    deleteTicket(index) {
      if (confirm('Are you sure you want to delete this ticket?')) {
        this.tickets.splice(index, 1);
        this.$toast?.info('Ticket deleted');
      }
    },
    deleteSelectedTickets() {
      if (confirm(`Are you sure you want to delete ${this.selectedTicketsCount} selected tickets?`)) {
        this.tickets = this.tickets.filter(ticket => !ticket.selected);
        this.$toast?.info(`${this.selectedTicketsCount} tickets deleted`);
      }
    },
    toggleTicketSelection(index, event) {
      this.$set(this.tickets[index], 'selected', event.target.checked);
    },
    selectTicket(index) {
      this.selectedTicketIndex = index;
    },
    closeModal() {
      this.showEditModal = false;
      this.editingTicketIndex = -1;
      this.editingTicket = this.getEmptyTicket();
    },
    handleNext() {
      if (this.tickets.length === 0) {
        this.$toast?.warning('Please create at least one ticket');
        return;
      }
      this.$emit('update:data', { 
        editedTickets: [...this.tickets]
      });
      this.$emit('next');
    }
  },
  mounted() {
    // Auto-generate tickets if we have transcription but no tickets yet
    if (this.data.transcription && (!this.data.tickets || this.data.tickets.length === 0)) {
      this.generateTickets();
    }
  }
};
</script>

<style scoped>
.ticket-item {
  transition: all 0.2s ease;
  cursor: pointer;
}

.ticket-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.1);
}

.ticket-item.border-primary {
  border-width: 2px;
}

.spin {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.modal {
  z-index: 1050;
  background-color: rgba(0, 0, 0, 0.5);
}

.modal-backdrop {
  z-index: 1040;
}

.ticket-list {
  max-height: 500px;
  overflow-y: auto;
  padding-right: 10px;
}

/* Custom scrollbar */
.ticket-list::-webkit-scrollbar {
  width: 8px;
}

.ticket-list::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

.ticket-list::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 4px;
}

.ticket-list::-webkit-scrollbar-thumb:hover {
  background: #555;
}
</style>
