<template>
  <div class="backlog-workflow">
    <!-- Breadcrumb Navigation -->
    <nav aria-label="breadcrumb">
      <ol class="breadcrumb">
        <li 
          v-for="(step, index) in steps" 
          :key="index"
          class="breadcrumb-item"
          :class="{ 
            'active': currentStep === index + 1,
            'text-primary': currentStep > index + 1,
            'cursor-pointer': currentStep > index + 1
          }"
          @click="navigateToStep(index + 1)"
        >
          <span v-if="currentStep > index + 1">
            <i class="bi bi-check-circle-fill text-success me-1"></i>
          </span>
          {{ step.title }}
        </li>
      </ol>
    </nav>

    <!-- Progress Bar -->
    <div class="progress mb-4">
      <div 
        class="progress-bar bg-primary" 
        role="progressbar" 
        :style="{ width: progressPercentage + '%' }"
        :aria-valuenow="progressPercentage" 
        aria-valuemin="0" 
        aria-valuemax="100"
      ></div>
    </div>

    <!-- Step Content -->
    <div class="workflow-content">
      <component 
        :is="currentComponent" 
        v-bind="currentProps"
        @next="nextStep"
        @prev="prevStep"
        @update:data="updateWorkflowData"
        @skip-upload="skipUpload"
      />
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import UploadStep from '@/components/workflow/UploadStep.vue';
import TextPreviewStep from '@/components/workflow/TextPreviewStep.vue';
import TicketCreationStep from '@/components/workflow/TicketCreationStep.vue';
import TicketEditStep from '@/components/workflow/TicketEditStep.vue';
import ProjectUploadStep from '@/components/workflow/ProjectUploadStep.vue';

export default {
  name: 'BacklogWorkflow',
  components: {
    UploadStep,
    TextPreviewStep,
    TicketCreationStep,
    TicketEditStep,
    ProjectUploadStep
  },
  setup() {
    // ...
    // skipUpload handler will be injected into returned object

    const currentStep = ref(1);
    const workflowData = ref({
      audioFile: null,
      transcription: '',
      tickets: [],
      editedTickets: [],
      selectedProject: null,
      selectedRepo: null
    });

    const steps = [
      { title: 'Upload Audio', component: 'UploadStep' },
      { title: 'Review Text', component: 'TextPreviewStep' },
      { title: 'Create Tickets', component: 'TicketCreationStep' },
      { title: 'Edit Tickets', component: 'TicketEditStep' },
      { title: 'Upload to Project', component: 'ProjectUploadStep' }
    ];

    const progressPercentage = computed(() => {
      return ((currentStep.value - 1) / (steps.length - 1)) * 100;
    });

    const currentComponent = computed(() => {
      return steps[currentStep.value - 1]?.component || 'UploadStep';
    });

    const currentProps = computed(() => {
      return {
        data: workflowData.value,
        currentStep: currentStep.value
      };
    });

    const skipUpload = () => {
      // Jump directly to Review Text step and clear audio file
      workflowData.value.audioFile = null;
      workflowData.value.transcription = '';
      currentStep.value = 2;
    };

    const nextStep = () => {
      if (currentStep.value < steps.length) {
        currentStep.value++;
      }
    };

    const prevStep = () => {
      if (currentStep.value > 1) {
        currentStep.value--;
      }
    };

    const navigateToStep = (step) => {
      if (step < currentStep.value) {
        currentStep.value = step;
      }
    };

    const updateWorkflowData = (newData) => {
      workflowData.value = { ...workflowData.value, ...newData };
    };

    return {
      currentStep,
      steps,
      progressPercentage,
      currentComponent,
      currentProps,
      nextStep,
      prevStep,
      updateWorkflowData,
      navigateToStep,
      skipUpload
    };
  }
};
</script>

<style scoped>
.breadcrumb {
  background-color: transparent;
  padding: 1rem 0;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid #eee;
}

.breadcrumb-item {
  font-size: 0.9rem;
  font-weight: 500;
  color: #6c757d;
}

.breadcrumb-item.active {
  color: #0d6efd;
  font-weight: 600;
}

.breadcrumb-item.cursor-pointer {
  cursor: pointer;
}

.workflow-content {
  min-height: 400px;
  padding: 1.5rem 0;
}

.progress {
  height: 8px;
  border-radius: 4px;
  background-color: #e9ecef;
  margin-bottom: 2rem;
}

.progress-bar {
  transition: width 0.3s ease;
}
</style>
