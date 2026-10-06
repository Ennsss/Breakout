window.breakout = {
  icons: () => window.lucide?.createIcons(),
  escape: (value) =>
    String(value ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    ),
  async api(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": document.querySelector('meta[name="csrf-token"]')
          .content,
        ...options.headers,
      },
    });
    const data = await response.json();
    if (!response.ok) {
      if (response.status === 401 && !url.endsWith("/login"))
        location.href = "/login";
      throw new Error(data.error || "Request failed. Please retry.");
    }
    return data;
  },
};
window.breakout.icons();
