<template>
  <div class="upload-step text-center">
    <h3 class="mb-4">Upload Audio/Video</h3>
    <p class="text-muted mb-4">Upload an audio or video file to transcribe it to text</p>
    
    <div 
      class="drop-zone p-5 mb-4 border rounded"
      @dragover.prevent="onDragOver"
      @dragleave="onDragLeave"
      @drop.prevent="onDrop"
      @click="triggerFileInput"
      :class="{ 'border-primary': isDragging }"
    >
      <i class="bi bi-cloud-upload fs-1 mb-3 text-primary"></i>
      <h5>Drag & drop your file here</h5>
      <p class="text-muted mb-0">or</p>
      <button class="btn btn-outline-primary mt-2">
        <i class="bi bi-upload me-2"></i>Browse Files
      </button>
      <input 
        type="file" 
        ref="fileInput" 
        class="d-none" 
        accept=".mp3,.wav,.mp4,.m4a,.ogg,.webm"
        @change="onFileSelected"
      >
    </div>

    <div v-if="file" class="file-info mb-4 p-3 bg-light rounded">
      <div class="d-flex justify-content-between align-items-center">
        <div>
          <i class="bi bi-file-earmark-music me-2"></i>
          {{ file.name }}
          <small class="text-muted d-block">
            {{ formatFileSize(file.size) }}
          </small>
        </div>
        <button class="btn btn-sm btn-outline-danger" @click="removeFile">
          <i class="bi bi-x"></i>
        </button>
      </div>
    </div>

    <div class="d-flex flex-column flex-md-row justify-content-between mt-4 gap-2">
      <button class="btn btn-outline-secondary" disabled>Previous</button>
      <button 
        class="btn btn-primary" 
        @click="handleNext"
        :disabled="!file || isLoading"
      >
        <span v-if="isLoading">
          <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
          Transcribing...
        </span>
        <span v-else>
          Next: Review Text <i class="bi bi-arrow-right ms-2"></i>
        </span>
      </button>
      <button class="btn btn-outline-info" @click="skipUpload">
        <i class="bi bi-arrow-right-circle me-2"></i>Skip Upload / Enter Text Manually
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'UploadStep',
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
  emits: ['next', 'update:data'],
  data() {
    return {
      isDragging: false,
      file: this.data.audioFile || null,
      isLoading: false
    };
  },
  methods: {
    skipUpload() {
      this.$emit('skip-upload');
    },
    onDragOver() {
      this.isDragging = true;
    },
    onDragLeave() {
      this.isDragging = false;
    },
    onDrop(event) {
      this.isDragging = false;
      const files = event.dataTransfer.files;
      if (files.length) {
        this.handleFile(files[0]);
      }
    },
    triggerFileInput() {
      this.$refs.fileInput.click();
    },
    onFileSelected(event) {
      const files = event.target.files;
      if (files.length) {
        this.handleFile(files[0]);
      }
    },
    handleFile(file) {
      const supportedTypes = ['audio/mp3', 'audio/wav', 'audio/mpeg', 'video/mp4', 'audio/ogg', 'audio/webm', 'audio/m4a'];
      if (!supportedTypes.includes(file.type)) {
        this.$toast.error('Unsupported file type. Please upload an audio or video file.');
        return;
      }
      
      this.file = file;
      this.$emit('update:data', { audioFile: file });
    },
    removeFile() {
      this.file = null;
      this.$refs.fileInput.value = '';
      this.$emit('update:data', { audioFile: null });
    },
    formatFileSize(bytes) {
      if (bytes === 0) return '0 Bytes';
      const k = 1024;
      const sizes = ['Bytes', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    async handleNext() {
      if (!this.file) return;
      
      this.isLoading = true;
      
      try {
        // Create FormData for file upload
        const formData = new FormData();
        formData.append('file', this.file);
        formData.append('language', 'en-US'); // Default language
        
        // Call the transcription API
        const response = await fetch('http://localhost:8000/transcribe', {
          method: 'POST',
          body: formData
        });
        
        if (!response.ok) {
          throw new Error('Transcription failed');
        }
        
        const result = await response.json();
        
        // Try different possible response structures
        const transcriptionText = result.text || result.transcription || result.transcript || result.content || '';
        
        if (!transcriptionText) {
          throw new Error('No transcription text in response');
        }
        
        // Update workflow data with transcription
        this.$emit('update:data', { 
          audioFile: this.file,
          transcription: transcriptionText
        });
        
        this.$toast?.success('Audio transcribed successfully');
        this.$emit('next');
        
      } catch (error) {
        console.error('Error during transcription:', error);
        this.$toast?.error('Failed to transcribe audio. Please try again.');
        
        // For demo purposes, proceed with mock data
        this.$emit('update:data', { 
          audioFile: this.file,
          transcription: 'Mock transcription: This is a sample transcription of the uploaded audio file. It contains various tasks and requirements that need to be converted into tickets.'
        });
        this.$emit('next');
      } finally {
        this.isLoading = false;
      }
    }
  }
};
</script>

<style scoped>
.drop-zone {
  border: 2px dashed #dee2e6;
  border-radius: 8px;
  transition: all 0.3s ease;
  cursor: pointer;
  background-color: #f8f9fa;
}

.drop-zone:hover,
.drop-zone.border-primary {
  border-color: #0d6efd;
  background-color: rgba(13, 110, 253, 0.05);
}

.file-info {
  border-left: 3px solid #0d6efd;
}

.btn-outline-primary {
  transition: all 0.2s ease;
}

.btn-outline-primary:hover {
  background-color: #0d6efd;
  color: white;
}
</style>
