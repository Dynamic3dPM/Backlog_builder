<template>
  <div class="upload-container">
    <h2 class="mb-4">Audio/Video to Text Conversion</h2>
    <p class="lead mb-4">
      Upload an audio or video file to transcribe it to text in real-time.
      Supported formats: MP3, WAV, MP4, M4A, OGG, WebM.
    </p>
    
    <div class="row">
      <div class="col-lg-10 mx-auto">
        <SpeechToText />
      </div>
    </div>
    
    <div class="mt-5">
      <h4>How It Works</h4>
      <ol class="list-group list-group-numbered">
        <li class="list-group-item">Upload an audio or video file by dragging and dropping or clicking to browse</li>
        <li class="list-group-item">Select the language of the audio content</li>
        <li class="list-group-item">Click "Start Transcription" to begin the conversion</li>
        <li class="list-group-item">View the transcribed text in real-time</li>
        <li class="list-group-item">Copy or download the transcription when complete</li>
      </ol>
    </div>
  </div>
</template>

<script>
import SpeechToText from '@/components/SpeechToText.vue';

export default {
  name: 'UploadView',
  components: {
    SpeechToText
  }
};
</script>

<style scoped>
.upload-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.lead {
  color: #6c757d;
  font-size: 1.1rem;
}

.list-group-item {
  border-left: none;
  border-right: none;
  padding: 1rem 1.5rem;
}

.list-group-item:first-child {
  border-top: none;
}

.list-group-item:last-child {
  border-bottom: none;
}
</style>