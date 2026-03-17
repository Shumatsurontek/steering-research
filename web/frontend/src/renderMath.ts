import katex from "katex";
import "katex/dist/katex.min.css";

/**
 * Render a text string containing LaTeX math ($...$, $$...$$, \boxed{})
 * and basic markdown (### headings, **bold**) into HTML.
 * Designed for streaming — handles partial/broken LaTeX gracefully.
 */
export function renderMathText(text: string): string {
  // Normalize \boxed{...} → LaTeX display blocks
  let s = text;

  // Replace $$...$$ (display math) first
  s = s.replace(/\$\$([^$]+?)\$\$/g, (_match, tex) => renderKatex(tex.trim(), true));

  // Replace $...$ (inline math) — avoid matching empty or multi-line
  s = s.replace(/\$([^$\n]+?)\$/g, (_match, tex) => renderKatex(tex.trim(), false));

  // ### headings → styled spans
  s = s.replace(/^### (.+)$/gm, '<div style="font-weight:600;font-size:0.9rem;margin:12px 0 4px;color:#0a0a0a">$1</div>');

  // **bold**
  s = s.replace(/\*\*([^*]+?)\*\*/g, "<strong>$1</strong>");

  return s;
}

function renderKatex(tex: string, displayMode: boolean): string {
  try {
    return katex.renderToString(tex, {
      displayMode,
      throwOnError: false,
      trust: true,
    });
  } catch {
    // Fallback: show raw LaTeX in a code span
    return `<code>${tex}</code>`;
  }
}
