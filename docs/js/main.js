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

/* ============================================================
   Floating mini-player for sleap_spatial.mp4
   - Appears after scrolling past the Introduction section
   - Draggable by header, resizable from all edges/corners
   - Fullscreen toggle and minimize/reopen
   ============================================================ */
(function () {
  'use strict';

  var player    = document.getElementById('mini-player');
  var video     = document.getElementById('mini-video');
  var header    = player && player.querySelector('.mini-player-header');
  var minimizeBtn = document.getElementById('mini-minimize');
  var reopenBtn   = document.getElementById('mini-reopen');
  var fsBtn       = document.getElementById('mini-fullscreen');

  if (!player || !video) return;

  var shown = false;
  var minimized = false;

  /* --- Show after scrolling past the hero/title screen --- */
  var heroSection = document.querySelector('section.hero');

  if (heroSection) {
    var heroObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting && entry.boundingClientRect.top < 0 && !minimized) {
          if (!shown) {
            shown = true;
            player.classList.remove('hidden');
            video.play();
          }
        }
      });
    }, { threshold: 0 });
    heroObserver.observe(heroSection);
  }

  /* --- Minimize: hide player, show reopen pill --- */
  minimizeBtn.addEventListener('click', function () {
    player.classList.add('hidden');
    video.pause();
    minimized = true;
    reopenBtn.classList.remove('hidden');
  });

  /* --- Reopen: show player, hide pill --- */
  reopenBtn.addEventListener('click', function () {
    minimized = false;
    reopenBtn.classList.add('hidden');
    player.classList.remove('hidden');
    // Reset position to bottom-right
    player.style.top = '';
    player.style.left = '';
    player.style.bottom = '20px';
    player.style.right = '20px';
    video.play();
  });

  /* --- Fullscreen toggle --- */
  fsBtn.addEventListener('click', function () {
    player.classList.toggle('fullscreen');
    if (player.classList.contains('fullscreen')) {
      player.style.top = '';
      player.style.left = '';
      player.style.bottom = '';
      player.style.right = '';
      player.style.width = '';
      player.style.height = '';
      video.muted = false;
    } else {
      player.style.top = '';
      player.style.left = '';
      player.style.bottom = '20px';
      player.style.right = '20px';
      player.style.width = '380px';
      player.style.height = '';
      video.muted = true;
    }
  });

  /* --- Dragging by header --- */
  var dragOffsetX = 0, dragOffsetY = 0, isDragging = false;

  header.addEventListener('mousedown', function (e) {
    if (player.classList.contains('fullscreen')) return;
    // Don't drag when clicking buttons
    if (e.target.closest('.mini-btn')) return;
    isDragging = true;
    player.classList.add('dragging');

    var rect = player.getBoundingClientRect();
    dragOffsetX = e.clientX - rect.left;
    dragOffsetY = e.clientY - rect.top;

    player.style.top = rect.top + 'px';
    player.style.left = rect.left + 'px';
    player.style.bottom = 'auto';
    player.style.right = 'auto';

    e.preventDefault();
  });

  document.addEventListener('mousemove', function (e) {
    if (!isDragging) return;
    var x = e.clientX - dragOffsetX;
    var y = e.clientY - dragOffsetY;
    x = Math.max(0, Math.min(x, window.innerWidth - player.offsetWidth));
    y = Math.max(0, Math.min(y, window.innerHeight - player.offsetHeight));
    player.style.left = x + 'px';
    player.style.top = y + 'px';
  });

  document.addEventListener('mouseup', function () {
    if (isDragging) {
      isDragging = false;
      player.classList.remove('dragging');
    }
  });

  /* --- Resizing from all edges and corners --- */
  var resizeDir = null;
  var resizeStartX, resizeStartY, resizeStartW, resizeStartH, resizeStartL, resizeStartT;

  var resizeHandles = player.querySelectorAll('.resize-handle');
  resizeHandles.forEach(function (handle) {
    handle.addEventListener('mousedown', function (e) {
      if (player.classList.contains('fullscreen')) return;
      resizeDir = this.dataset.dir;
      player.classList.add('dragging');

      var rect = player.getBoundingClientRect();
      resizeStartX = e.clientX;
      resizeStartY = e.clientY;
      resizeStartW = rect.width;
      resizeStartH = rect.height;
      resizeStartL = rect.left;
      resizeStartT = rect.top;

      // Switch to top/left positioning
      player.style.top = rect.top + 'px';
      player.style.left = rect.left + 'px';
      player.style.bottom = 'auto';
      player.style.right = 'auto';

      e.preventDefault();
      e.stopPropagation();
    });
  });

  document.addEventListener('mousemove', function (e) {
    if (!resizeDir) return;

    var dx = e.clientX - resizeStartX;
    var dy = e.clientY - resizeStartY;
    var minW = 200, minH = 140;

    var newW = resizeStartW;
    var newH = resizeStartH;
    var newL = resizeStartL;
    var newT = resizeStartT;

    // East edge: grow width rightward
    if (resizeDir.indexOf('e') !== -1) {
      newW = Math.max(minW, resizeStartW + dx);
    }
    // West edge: grow width leftward, shift left
    if (resizeDir.indexOf('w') !== -1) {
      newW = Math.max(minW, resizeStartW - dx);
      newL = resizeStartL + (resizeStartW - newW);
    }
    // South edge: grow height downward
    if (resizeDir.indexOf('s') !== -1) {
      newH = Math.max(minH, resizeStartH + dy);
    }
    // North edge: grow height upward, shift top
    if (resizeDir.indexOf('n') !== -1) {
      newH = Math.max(minH, resizeStartH - dy);
      newT = resizeStartT + (resizeStartH - newH);
    }

    // Clamp to viewport
    newL = Math.max(0, newL);
    newT = Math.max(0, newT);

    player.style.width = newW + 'px';
    player.style.height = newH + 'px';
    player.style.left = newL + 'px';
    player.style.top = newT + 'px';
  });

  document.addEventListener('mouseup', function () {
    if (resizeDir) {
      resizeDir = null;
      player.classList.remove('dragging');
    }
  });
})();
