(function () {
  var D = window.TRAFFIC; if (!D) return;
  var CYCLE = 90, GREEN_TOTAL = 80, AMBER = 5, MING = 10, SAT = 1800, FIXED = 40;
  var NAMES = ["North", "South", "East", "West"];
  var $ = function (id) { return document.getElementById(id); };

  function predict(v) {
    var x = v.map(function (a) { return a / D.model.input_scale; });
    var W = D.model.weights, B = D.model.biases;
    for (var l = 0; l < W.length; l++) {
      var out = B[l].slice();
      for (var j = 0; j < out.length; j++) for (var i = 0; i < x.length; i++) out[j] += x[i] * W[l][i][j];
      x = l < W.length - 1 ? out.map(function (a) { return Math.max(0, a); }) : out;
    }
    return Math.round(Math.min(Math.max(x[0] * D.model.output_scale, MING), GREEN_TOTAL - MING));
  }
  function sat(v, g) { return v / (SAT * g / CYCLE); }
  function pct(x) { return Math.round(x * 100) + "%"; }

  var sliders = NAMES.map(function (n) { return $("v-" + n.toLowerCase()); });
  var hourEl = $("hour");

  function fmtHour(h) {
    var hh = Math.floor(h), mm = (h - hh) >= 0.5 ? "30" : "00";
    return (hh < 10 ? "0" : "") + hh + ":" + mm;
  }
  function cycleBar(el, g, fixed) {
    var ew = GREEN_TOTAL - g;
    el.className = "cyc" + (fixed ? " fixed" : "");
    el.innerHTML = '<span class="ns" style="flex:' + g + '">N–S ' + g + 's</span><span class="am" style="flex:' + AMBER + '"></span>' +
      '<span class="ew" style="flex:' + ew + '">E–W ' + ew + 's</span><span class="am" style="flex:' + AMBER + '"></span>';
  }
  function meter(x, fixed) {
    var over = x > 1;
    return '<div class="meter' + (fixed ? " fx" : "") + (over ? " over" : "") + '"><i style="width:' + Math.min(x, 1.25) / 1.25 * 100 + '%"></i><b>' + pct(x) + '</b></div>';
  }
  function update() {
    var v = sliders.map(function (s) { return +s.value; });
    sliders.forEach(function (s, i) { $("o-" + NAMES[i].toLowerCase()).textContent = s.value + " veh/h"; });
    var g = predict(v);
    cycleBar($("bar-ai"), g, false);
    cycleBar($("bar-fixed"), FIXED, true);
    $("tag-ai").innerHTML = "<span>AI recommendation</span><span>cycle " + CYCLE + "s</span>";
    var rows = '<span class="hd"></span><span class="hd">Fixed 40/40</span><span class="hd">AI plan</span>';
    for (var i = 0; i < 4; i++) {
      var gi = i < 2 ? g : GREEN_TOTAL - g;
      var gf = FIXED;
      rows += "<span>" + NAMES[i] + "</span>" + meter(sat(v[i], gf), true) + meter(sat(v[i], gi), false);
    }
    $("sat").innerHTML = rows;
    var worstF = Math.max.apply(null, v.map(function (a) { return sat(a, FIXED); }));
    var worstA = Math.max.apply(null, v.map(function (a, i) { return sat(a, i < 2 ? g : GREEN_TOTAL - g); }));
    $("summary").textContent = "Busiest approach is at " + pct(worstF) + " of capacity with the fixed plan and " + pct(worstA) + " with the AI plan." +
      (worstF > 1 && worstA <= 1 ? " The fixed plan cannot clear that queue; the AI plan can." : "");
  }
  function setHour(h, preset) {
    var row = D.profile[Math.floor(h) % 24];
    sliders.forEach(function (s, i) { s.value = Math.round(row[i] / 10) * 10; });
    $("clock").firstChild.nodeValue = fmtHour(h);
    update();
  }
  hourEl.addEventListener("input", function () {
    clearPresetState(); setHour(+hourEl.value);
  });
  sliders.forEach(function (s) {
    s.addEventListener("input", function () { clearPresetState(); $("clock-note").textContent = "custom volumes"; update(); });
  });
  function clearPresetState() {
    Array.prototype.forEach.call(document.querySelectorAll(".presets button"), function (b) { b.setAttribute("aria-pressed", "false"); });
    $("clock-note").textContent = "volumes follow the simulated day";
  }
  Array.prototype.forEach.call(document.querySelectorAll(".presets button"), function (b) {
    b.addEventListener("click", function () {
      var h = +b.dataset.h; hourEl.value = h; setHour(h);
      clearPresetState(); b.setAttribute("aria-pressed", "true");
    });
  });
  setHour(+hourEl.value);
})();
