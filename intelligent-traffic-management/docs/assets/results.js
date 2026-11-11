(function () {
  var D = window.TRAFFIC; if (!D) return; var R = D.results;
  function row(k) {
    var r = R[k];
    return "<tr><td>" + r.name + '</td><td class="num">' + r.avg_delay_s.toFixed(1) + " ± " + r.avg_delay_sd.toFixed(1) +
      '</td><td class="num">' + r.peak_delay_s.toFixed(1) + '</td><td class="num">' + r.max_queue.toFixed(1) +
      '</td><td class="num">' + r.mean_queue.toFixed(1) + "</td></tr>";
  }
  var t = document.getElementById("results-body");
  if (t) t.innerHTML = row("fixed") + row("ai") + row("rule");
  var set = function (id, v) { var e = document.getElementById(id); if (e) e.textContent = v; };
  set("imp-avg", R.improvement_avg_pct + "%");
  set("imp-peak", R.improvement_peak_pct + "%");
  set("veh", R.vehicles.toLocaleString("en-GB"));
  var m = D.model.metrics;
  set("mae", m.mae_seconds + " s");
  set("mae-fixed", m.fixed_plan_mae_seconds + " s");
  set("r2", m.r2);
})();
