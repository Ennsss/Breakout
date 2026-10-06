const form = document.querySelector("#auth-form");
document.querySelector("#show-password").onclick = (event) => {
  const input = document.querySelector("#password");
  const visible = input.type === "password";
  input.type = visible ? "text" : "password";
  const button = event.currentTarget;
  button.setAttribute(
    "aria-label",
    visible ? "Hide password" : "Show password",
  );
  button.title = button.getAttribute("aria-label");
  button.innerHTML = `<i data-lucide="${visible ? "eye-off" : "eye"}"></i>`;
  breakout.icons();
};
form.onsubmit = async (event) => {
  event.preventDefault();
  const button = document.querySelector("#auth-submit");
  const error = document.querySelector("#auth-error");
  button.disabled = true;
  error.hidden = true;
  try {
    await breakout.api(`/api/${form.dataset.mode}`, {
      method: "POST",
      body: JSON.stringify(Object.fromEntries(new FormData(form))),
    });
    location.href = "/app";
  } catch (e) {
    error.textContent = e.message;
    error.hidden = false;
    button.disabled = false;
  }
};
