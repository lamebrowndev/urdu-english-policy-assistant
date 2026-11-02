document.addEventListener("DOMContentLoaded", function () {
  var forms = document.querySelectorAll("form[data-busy]");
  for (var i = 0; i < forms.length; i += 1) {
    forms[i].addEventListener("submit", function (event) {
      var button = event.currentTarget.querySelector("button[data-busy-label]");
      if (button) {
        button.disabled = true;
        button.textContent = button.getAttribute("data-busy-label");
      }
    });
  }
});
