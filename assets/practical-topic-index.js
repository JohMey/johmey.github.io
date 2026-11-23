(() => {
  "use strict";

  const greek = {α:"alpha",β:"beta",γ:"gamma",δ:"delta",ε:"epsilon",ζ:"zeta",η:"eta",θ:"theta",
    ι:"iota",κ:"kappa",λ:"lambda",μ:"mu",ν:"nu",ξ:"xi",ο:"omicron",π:"pi",ρ:"rho",σ:"sigma",
    τ:"tau",υ:"upsilon",φ:"phi",χ:"chi",ψ:"psi",ω:"omega",ς:"sigma"};
  function normalise(s) {
    return s.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase()
      .replace(/[α-ω]/g, char => " " + greek[char] + " ").replace(/[^a-z0-9]+/g, " ").trim();
  }

  const pageSize = 10;
  for (const panel of document.querySelectorAll(".topic-search")) {
    const entries = JSON.parse(panel.querySelector("[data-topic-entries]").textContent);
    const topics = [];
    for (const [group, url, chapter, introduction, sections] of entries) {
      topics.push({url, label: chapter, context: group, heading: normalise(chapter), description: normalise(introduction)});
      for (const [id, title] of sections) {
        topics.push({url: url + "#" + id, label: title, context: chapter, heading: normalise(title), description: normalise(chapter), section:true});
      }
    }
    const input = panel.querySelector('input[type="search"]');
    const results = panel.querySelector(".topic-results");
    const count = panel.querySelector(".topic-count");
    const more = panel.querySelector("[data-more]");
    let matches = [];
    let shown = 0;

    function showNext() {
      for (const entry of matches.slice(shown, shown + pageSize)) {
        const row = document.createElement("li");
        const link = document.createElement("a");
        link.href = entry.url;
        link.textContent = entry.label;
        const group = document.createElement("small");
        group.textContent = entry.context;
        row.append(link, group);
        results.append(row);
      }
      shown = results.childElementCount;
      more.hidden = shown >= matches.length;
      count.textContent = matches.length
        ? shown + " of " + matches.length + (matches.length === 1 ? " match." : " matches.")
        : "No matches. Try a shorter term or browse the contents below.";
    }

    function search() {
      const phrase = normalise(input.value);
      results.replaceChildren();
      matches = [];
      shown = 0;
      more.hidden = true;
      count.textContent = "";
      if (!phrase) return;
      const terms = phrase.split(" ");
      for (const t of topics) {
        const inHeading = terms.filter(term => t.heading.includes(term)).length;
        if (!terms.every(term => t.heading.includes(term) || t.description.includes(term))) continue;
        if (!inHeading && t.section) continue;
        const score = inHeading + (t.heading.includes(phrase) ? 3 : 0) + (t.heading === phrase ? 5 : 0);
        matches.push({...t, score});
      }
      matches.sort((a, b) => b.score - a.score);
      showNext();
    }
    input.addEventListener("input", search);
    more.addEventListener("click", showNext);
    panel.querySelector("[data-topic-controls]").hidden = false;
    search();
  }
})();
