<template>
  <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
    <div class="container">
      <router-link class="navbar-brand" to="/">
        <i class="bi bi-clipboard-data me-2"></i>
        Backlog Builder
      </router-link>
      
      <button 
        class="navbar-toggler" 
        type="button" 
        data-bs-toggle="collapse" 
        data-bs-target="#navbarNav"
        aria-controls="navbarNav" 
        aria-expanded="false" 
        aria-label="Toggle navigation"
      >
        <span class="navbar-toggler-icon"></span>
      </button>
      
      <div class="collapse navbar-collapse" id="navbarNav">
        <ul class="navbar-nav me-auto">
          <li class="nav-item">
            <router-link 
              class="nav-link" 
              to="/workflow"
              :class="{ active: $route.path === '/workflow' }"
            >
              <i class="bi bi-diagram-3 me-1"></i>
              Workflow
            </router-link>
          </li>
        </ul>

        <ul class="navbar-nav">
          <li class="nav-item dropdown">
            <a 
              class="nav-link dropdown-toggle" 
              href="#" 
              id="navbarDropdown" 
              role="button" 
              data-bs-toggle="dropdown" 
              aria-expanded="false"
            >
              <i class="bi bi-person-circle me-1"></i>
              Account
            </a>
            <ul class="dropdown-menu dropdown-menu-end">
              <li><a class="dropdown-item" href="#"><i class="bi bi-gear me-2"></i>Settings</a></li>
              <li><a class="dropdown-item" href="#"><i class="bi bi-question-circle me-2"></i>Help</a></li>
              <li><hr class="dropdown-divider"></li>
              <li><a class="dropdown-item" href="#"><i class="bi bi-box-arrow-right me-2"></i>Logout</a></li>
            </ul>
          </li>
        </ul>
      </div>
    </div>
  </nav>
</template>

<script>
export default {
  name: 'AppNavigation'
};
</script>

<style scoped>
.navbar-brand {
  font-weight: 600;
  font-size: 1.25rem;
}

.nav-link {
  font-weight: 500;
  transition: all 0.2s ease;
}

.nav-link:hover {
  background-color: rgba(255, 255, 255, 0.1);
  border-radius: 0.375rem;
}

.nav-link.active {
  background-color: rgba(255, 255, 255, 0.2);
  border-radius: 0.375rem;
}

.dropdown-menu {
  border: none;
  box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
}

.dropdown-item {
  transition: all 0.2s ease;
}

.dropdown-item:hover {
  background-color: #f8f9fa;
}
</style>
