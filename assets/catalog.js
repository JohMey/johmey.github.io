const search = document.querySelector("#case-search");
const status = document.querySelector("#case-count");
const sections = [...document.querySelectorAll(".catalog-group")];
const entries = new Map(sections.map((section) => [
  section,
  [...section.querySelectorAll(".case-list > li")].map((entry) => ({
    entry,
    text: entry.textContent.toLowerCase(),
  })),
]));

function filterCases() {
  const terms = search.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
  let total = 0;
  for (const [section, items] of entries) {
    let count = 0;
    for (const {entry, text} of items) {
      entry.hidden = !terms.every((term) => text.includes(term));
      if (!entry.hidden) count++;
    }
    section.hidden = count === 0;
    total += count;
  }
  status.textContent = total === 1 ? "1 example" : `${total} examples`;
}

search.addEventListener("input", filterCases);
search.closest(".catalog-search").hidden = false;
filterCases();
