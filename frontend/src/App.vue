<template>
  <div id="app">
    <AppNavigation />
    
    <div class="container mt-4">
      <router-view></router-view>
    </div>
  </div>
</template>

<script>
import AppNavigation from './components/AppNavigation.vue';

export default {
  name: 'App',
  components: {
    AppNavigation
  }
};
</script>

<style>
#app {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
}

.navbar {
  margin-bottom: 2rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.navbar-brand {
  font-weight: 600;
}

.nav-link {
  font-weight: 500;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 15px;
}

.mt-4 {
  margin-top: 1.5rem !important;
}
</style>