<template>
  <div class="text-preview-step">
    <h3 class="mb-4">Review Transcription</h3>
    <p class="text-muted mb-4">Review and edit the transcribed text before creating tickets</p>
    
    <div class="card mb-4">
      <div class="card-header d-flex justify-content-between align-items-center">
        <span>Transcription</span>
        <div>
          <button class="btn btn-sm btn-outline-secondary me-2" @click="copyToClipboard">
            <i class="bi bi-clipboard me-1"></i> Copy
          </button>
          <button class="btn btn-sm btn-outline-secondary" @click="downloadText">
            <i class="bi bi-download me-1"></i> Download
          </button>
        </div>
      </div>
      <div class="card-body">
        <div v-if="!editedText.trim()" class="alert alert-info mb-3">
          <i class="bi bi-info-circle me-2"></i>
          No transcription loaded. <b>You may paste or type your own meeting notes or transcript below.</b>
        </div>
        <textarea 
          class="form-control" 
          v-model="editedText" 
          rows="12"
          placeholder="Paste or type your meeting transcript here..."
        ></textarea>
      </div>
    </div>

    <div class="d-flex justify-content-between mt-4">
      <button class="btn btn-outline-secondary" @click="$emit('prev')">
        <i class="bi bi-arrow-left me-2"></i> Back
      </button>
      <button 
        class="btn btn-primary" 
        @click="handleNext"
        :disabled="!editedText.trim()"
      >
        Next: Create Tickets <i class="bi bi-arrow-right ms-2"></i>
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'TextPreviewStep',
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
      editedText: this.data.transcription || '',
      isLoading: false
    };
  },
  mounted() {
    // Component is ready
  },
  watch: {
    editedText(newText) {
      this.$emit('update:data', { transcription: newText });
    },
    'data.transcription'(newTranscription) {
      if (newTranscription && newTranscription !== this.editedText) {
        this.editedText = newTranscription;
      }
    }
  },
  methods: {
    copyToClipboard() {
      navigator.clipboard.writeText(this.editedText);
      this.$toast.success('Text copied to clipboard');
    },
    downloadText() {
      const blob = new Blob([this.editedText], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'transcription.txt';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    },
    handleNext() {
      this.$emit('next');
    }
  }
};
</script>

<style scoped>
.text-preview-step {
  max-width: 800px;
  margin: 0 auto;
}

textarea {
  min-height: 300px;
  resize: vertical;
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

.btn-outline-secondary {
  transition: all 0.2s ease;
}

.btn-outline-secondary:hover {
  background-color: #6c757d;
  color: white;
}
</style>
