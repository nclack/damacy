(() => {
  const key = "damacy-appearance";
  const system = matchMedia("(prefers-color-scheme: dark)");
  let preference = "system";
  try {
    const saved = localStorage.getItem(key);
    if (["system", "light", "dark"].includes(saved)) preference = saved;
  } catch {}
  const apply = () => {
    document.documentElement.dataset.theme = preference === "system"
      ? (system.matches ? "dark" : "light") : preference;
  };
  apply();
  system.addEventListener("change", apply);
  document.addEventListener("DOMContentLoaded", () => {
    const select = document.getElementById("appearance");
    select.value = preference;
    select.addEventListener("change", () => {
      preference = select.value;
      try { localStorage.setItem(key, preference); } catch {}
      apply();
    });
  });
})();
