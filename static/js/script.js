/* =========================================================
   DOM REFERENCES
========================================================= */
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const previewContainer = document.getElementById("preview-container");
const previewImage = document.getElementById("preview-image");
const topKInput = document.getElementById("topK");
const rangeValue = document.getElementById("rangeValue");
const searchForm = document.getElementById("searchForm");
const submitBtn = document.getElementById("submitBtn");
const resetBtn = document.getElementById("resetBtn");

/* =========================================================
   DRAG & DROP + CLICK UPLOAD
========================================================= */

// Click to open file picker
dropzone.addEventListener("click", () => {
    fileInput.click();
});

// Drag over
dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});

// Drag leave
dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

// Drop file (SAFE method)
dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");

    const files = e.dataTransfer.files;
    if (!files || files.length === 0) return;

    // Browser-safe FileList assignment
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(files[0]);
    fileInput.files = dataTransfer.files;

    handlePreview();
});

// File selected via picker
fileInput.addEventListener("change", handlePreview);

/* =========================================================
   FILE VALIDATION + PREVIEW
========================================================= */

function handlePreview() {
    const file = fileInput.files[0];
    if (!file) return;

    // Type validation
    if (!file.type.startsWith("image/")) {
        alert("Only image files are allowed.");
        resetForm();
        return;
    }

    // Size validation (16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert("File size must be less than 16MB.");
        resetForm();
        return;
    }

    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = "block";
    };
    reader.readAsDataURL(file);

    // Update dropzone text
    dropzone.querySelector(".dropzone-text").innerHTML =
        `<span style="color:#66dd99;">✓ ${file.name}</span>`;
}

/* =========================================================
   SLIDER
========================================================= */

window.updateRange = function (val) {
    rangeValue.innerText = val;
};

// Initialize slider text
if (topKInput && rangeValue) {
    rangeValue.innerText = topKInput.value;
}

/* =========================================================
   FORM SUBMISSION
========================================================= */

searchForm.addEventListener("submit", () => {
    // Prevent double submit
    submitBtn.disabled = true;
    submitBtn.innerHTML = "Searching...";
});

/* =========================================================
   RESET FORM
========================================================= */

resetBtn.addEventListener("click", () => {
    resetForm();
});

function resetForm() {
    fileInput.value = "";
    previewContainer.style.display = "none";
    previewImage.src = "";
    topKInput.value = 10;
    rangeValue.innerText = "10";

    dropzone.querySelector(".dropzone-text").innerHTML =
        `<span class="dropzone-highlight">Click to upload</span> or drag & drop`;

    submitBtn.disabled = false;
    submitBtn.innerHTML = "Search Now";
}
