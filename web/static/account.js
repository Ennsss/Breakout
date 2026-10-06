const error = document.querySelector('#account-error');
async function updateAccount(url, body, button) {
  button.disabled = true;
  error.hidden = true;
  try {
    await window.breakout.api(url, {method: 'POST', body: JSON.stringify(body)});
    document.querySelector('#account-status').textContent = 'Done. Please sign in again.';
    location.href = '/login';
  } catch (failure) {
    error.textContent = failure.message;
    error.hidden = false;
    button.disabled = false;
  }
}
document.querySelector('#password-form').addEventListener('submit', event => {
  event.preventDefault();
  updateAccount('/api/password', Object.fromEntries(new FormData(event.target)), event.submitter);
});
document.querySelector('#logout-all').addEventListener('click', event => updateAccount('/api/logout-all', {}, event.currentTarget));
