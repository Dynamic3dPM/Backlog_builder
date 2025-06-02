<template>
  <div class="speech-to-text">
    <div class="card mb-4">
      <div class="card-header bg-primary text-white">
        <h5 class="mb-0">Audio/Video to Text</h5>
      </div>
      <div class="card-body">
        <div 
          class="drop-zone p-5 text-center border rounded mb-3"
          @dragover.prevent="onDragOver"
          @dragleave="onDragLeave"
          @drop.prevent="onDrop"
          @click="triggerFileInput"
          :class="{ 'border-primary': isDragging }"
        >
          <i class="bi bi-cloud-upload fs-1 mb-2"></i>
          <p class="mb-0">
            Drag & drop your audio/video file here<br>or<br>
            <span class="text-primary">Click to browse files</span>
          </p>
          <input 
            type="file" 
            ref="fileInput" 
            class="d-none" 
            accept=".mp3,.wav,.mp4,.m4a,.ogg,.webm"
            @change="onFileSelected"
          >
        </div>
        
        <div v-if="selectedFile" class="file-info mb-3 p-3 bg-light rounded">
          <div class="d-flex justify-content-between align-items-center">
            <div>
              <i class="bi bi-file-earmark-music me-2"></i>
              {{ selectedFile.name }}
              <small class="text-muted d-block">
                {{ formatFileSize(selectedFile.size) }}
              </small>
            </div>
            <button class="btn btn-sm btn-outline-danger" @click="removeFile">
              <i class="bi bi-x"></i>
            </button>
          </div>
        </div>

        <div class="language-selector mb-3">
          <label for="language" class="form-label">Language:</label>
          <select 
            id="language" 
            class="form-select" 
            v-model="selectedLanguage"
            :disabled="isTranscribing"
          >
            <option v-for="lang in languages" :key="lang.code" :value="lang.code">
              {{ lang.name }}
            </option>
          </select>
        </div>

        <div class="progress mb-3" v-if="isTranscribing">
          <div 
            class="progress-bar progress-bar-striped progress-bar-animated" 
            role="progressbar" 
            :style="{ width: progress + '%' }"
            :aria-valuenow="progress" 
            aria-valuemin="0" 
            aria-valuemax="100"
          >
            {{ progress }}%
          </div>
        </div>

        <div class="d-grid gap-2 d-md-flex justify-content-md-start mb-3">
          <button 
            class="btn btn-primary me-md-2"
            @click="startTranscription"
            :disabled="!selectedFile || isTranscribing"
          >
            <i class="bi bi-mic me-1"></i>
            {{ isTranscribing ? 'Transcribing...' : 'Start Transcription' }}
          </button>
          <button 
            class="btn btn-outline-secondary"
            @click="stopTranscription"
            :disabled="!isTranscribing"
          >
            <i class="bi bi-stop-fill me-1"></i>
            Stop
          </button>
        </div>

        <div class="transcription-output mb-3">
          <label class="form-label">Transcription:</label>
          <div 
            class="form-control" 
            style="min-height: 200px; max-height: 400px; overflow-y: auto;"
            contenteditable
            @input="onTextChange"
            ref="transcriptionOutput"
          ></div>
        </div>

        <div class="d-grid gap-2 d-md-flex justify-content-md-end">
          <button 
            class="btn btn-outline-secondary me-md-2"
            @click="copyToClipboard"
            :disabled="!transcriptionText"
          >
            <i class="bi bi-clipboard me-1"></i>
            Copy Text
          </button>
          <button 
            class="btn btn-primary"
            @click="downloadText"
            :disabled="!transcriptionText"
          >
            <i class="bi bi-download me-1"></i>
            Download as TXT
          </button>
        </div>
      </div>
    </div>
  </div>
</template>


<script>
import { getCurrentInstance } from 'vue';

export default {
  name: 'SpeechToText',
  setup() {
    // For now, use the direct URL to the whisper-stt service
    // In production, you might want to use a proxy or environment variable
    const apiBaseUrl = 'http://localhost:8000/transcribe';
    
    // Log the API URL for debugging
    console.log('Using API URL:', apiBaseUrl);
    
    return { apiBaseUrl };
  },
  data() {
    return {
      isDragging: false,
      selectedFile: null,
      isTranscribing: false,
      progress: 0,
      transcriptionText: '',
      selectedLanguage: 'en-US',
      recognition: null,
      languages: [
        { code: 'en-US', name: 'English (US)' },
        { code: 'es-ES', name: 'Spanish' },
        { code: 'fr-FR', name: 'French' },
        { code: 'de-DE', name: 'German' },
        { code: 'it-IT', name: 'Italian' },
        { code: 'pt-BR', name: 'Portuguese' },
        { code: 'ru-RU', name: 'Russian' },
        { code: 'zh-CN', name: 'Chinese' },
        { code: 'ja-JP', name: 'Japanese' },
        { code: 'ko-KR', name: 'Korean' }
      ]
    };
  },
  methods: {
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
      // Check if file type is supported
      const supportedTypes = ['audio/mp3', 'audio/wav', 'audio/mpeg', 'video/mp4', 'audio/ogg', 'audio/webm', 'audio/m4a'];
      if (!supportedTypes.includes(file.type)) {
        this.$toast.error('Unsupported file type. Please upload an audio or video file.');
        return;
      }
      
      this.selectedFile = file;
      this.transcriptionText = '';
      this.$refs.transcriptionOutput.textContent = '';
    },
    removeFile() {
      this.selectedFile = null;
      this.$refs.fileInput.value = '';
      this.transcriptionText = '';
      this.$refs.transcriptionOutput.textContent = '';
    },
    formatFileSize(bytes) {
      if (bytes === 0) return '0 Bytes';
      const k = 1024;
      const sizes = ['Bytes', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    async startTranscription() {
      if (!this.selectedFile) return;
      
      this.isTranscribing = true;
      this.progress = 0;
      this.transcriptionText = '';
      this.$refs.transcriptionOutput.textContent = 'Uploading and processing...';
      
      try {
        const formData = new FormData();
        formData.append('file', this.selectedFile);
        formData.append('language', this.selectedLanguage.split('-')[0]); // Convert en-US to en
        formData.append('task', 'transcribe');
        formData.append('return_timestamps', 'true');
        
        // Show progress for file upload
        const progressInterval = setInterval(() => {
          if (this.progress < 90) {
            this.progress += 5;
          }
        }, 300);
        
        const response = await this.$http({
          method: 'post',
          url: this.apiBaseUrl,  // Use the full URL with the endpoint
          data: formData,
          headers: {
            'Content-Type': 'multipart/form-data',
            'Accept': 'application/json'
          },
          onUploadProgress: (progressEvent) => {
            if (progressEvent.lengthComputable) {
              const percentCompleted = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
              this.progress = Math.min(percentCompleted, 90);
            }
          },
        });
        
        clearInterval(progressInterval);
        this.progress = 100;
        
        // Handle the response from the Whisper STT service
        if (response.data) {
          // Check if the response contains segments (timestamps)
          if (response.data.segments && response.data.segments.length > 0) {
            // Join all text segments with newlines
            this.transcriptionText = response.data.segments
              .map(segment => segment.text.trim())
              .join('\n\n');
          } else if (response.data.text) {
            // Fallback to text if no segments
            this.transcriptionText = response.data.text;
          } else {
            this.transcriptionText = 'No transcript available';
          }
          
          this.$refs.transcriptionOutput.textContent = this.transcriptionText;
          
          // Show success message
          const app = getCurrentInstance().appContext.app;
          app.config.globalProperties.$toast.success('Transcription completed successfully!');
        } else {
          throw new Error('No transcription result received');
        }
      } catch (error) {
        console.error('Transcription error:', error);
        let errorMessage = 'Error during transcription';
        
        if (error.response) {
          // The request was made and the server responded with a status code
          // that falls out of the range of 2xx
          console.error('Response data:', error.response.data);
          console.error('Response status:', error.response.status);
          console.error('Response headers:', error.response.headers);
          
          errorMessage = `Server error: ${error.response.status} - ${error.response.data?.message || 'Unknown error'}`;
        } else if (error.request) {
          // The request was made but no response was received
          console.error('Request was made but no response received:', error.request);
          errorMessage = 'No response from server. Please check if the service is running.';
        } else {
          // Something happened in setting up the request that triggered an Error
          console.error('Error setting up request:', error.message);
          errorMessage = `Request error: ${error.message}`;
        }
        
        try {
          // Use the toast plugin directly from the app instance
          const app = getCurrentInstance().appContext.app;
          app.config.globalProperties.$toast.error(errorMessage);
        } catch (e) {
          console.error('Failed to show error toast:', e);
        }
      } finally {
        this.isTranscribing = false;
      }
    },
    stopTranscription() {
      this.isTranscribing = false;
      this.progress = 0;
    },
    onTextChange(event) {
      this.transcriptionText = event.target.innerText;
    },
    copyToClipboard() {
      navigator.clipboard.writeText(this.transcriptionText).then(() => {
        this.$toast.success('Text copied to clipboard!');
      }).catch(err => {
        console.error('Failed to copy text: ', err);
        this.$toast.error('Failed to copy text');
      });
    },
    downloadText() {
      const element = document.createElement('a');
      const file = new Blob([this.transcriptionText], { type: 'text/plain' });
      element.href = URL.createObjectURL(file);
      const fileName = this.selectedFile ? 
        this.selectedFile.name.replace(/\.[^/.]+$/, '') + '.txt' : 'transcription.txt';
      element.download = fileName;
      document.body.appendChild(element);
      element.click();
      document.body.removeChild(element);
    }
  },
  beforeUnmount() {
    this.stopTranscription();
  }
};
</script>

<style scoped>
.drop-zone {
  border: 2px dashed #adb5bd;
  border-radius: 0.375rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.drop-zone:hover {
  border-color: #0d6efd;
  background-color: rgba(13, 110, 253, 0.05);
}

.transcription-output {
  white-space: pre-wrap;
  word-break: break-word;
  min-height: 200px;
  border: 1px solid #dee2e6;
  border-radius: 0.375rem;
  padding: 0.5rem;
  overflow-y: auto;
}

.progress {
  height: 1.5rem;
  font-size: 0.875rem;
}

.progress-bar {
  transition: width 0.3s ease;
}
</style>
