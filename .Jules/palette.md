
## 2024-05-18 - Added Accessibility Attributes to Dashboard Mobile Menu
**Learning:** Icon-only buttons for mobile navigation lack readable names, and dropdown menus often miss ARIA associations with their toggles. Form labels in typical layouts often lack `htmlFor` IDs linking them to their corresponding inputs.
**Action:** Always add `aria-label`, `aria-expanded`, and `aria-controls` to mobile menu toggle buttons. Explicitly link `<label>` and `<input>` using `htmlFor` and `id`.
