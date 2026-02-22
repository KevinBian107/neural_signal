/* Scroll-reveal animation using IntersectionObserver */
(function () {
  'use strict';

  var elements = document.querySelectorAll('.reveal');

  if (!('IntersectionObserver' in window)) {
    // Fallback: show everything immediately
    elements.forEach(function (el) { el.classList.add('visible'); });
    return;
  }

  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.1,
    rootMargin: '0px 0px -40px 0px'
  });

  elements.forEach(function (el) { observer.observe(el); });
})();
