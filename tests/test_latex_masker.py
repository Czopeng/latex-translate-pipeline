"""Unit tests for LaTeX masking and unmasking behavior."""

from __future__ import annotations

import unittest

from translator.latex_masker import LatexMasker


class LatexMaskerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.masker = LatexMasker()

    def test_round_trip_preserves_original_text(self) -> None:
        source = (
            "Dies ist ein Test mit \\gls{WBG}, \\cite{ref:key}, "
            "und \\ref{sec:intro}."
        )
        result = self.masker.mask_latex_commands(source)
        restored = self.masker.restore_latex_commands(result.masked_text, result.placeholder_map)

        self.assertEqual(source, restored)

    def test_placeholders_are_generated(self) -> None:
        source = "Messung mit \\textbf{Signal} und \\% Restfehler."
        result = self.masker.mask_latex_commands(source)

        self.assertIn("__CMD1__", result.masked_text)
        self.assertGreaterEqual(len(result.placeholder_map), 2)

    def test_collision_safe_placeholder_generation(self) -> None:
        # Literal token-like content must not be overwritten by generated placeholders.
        source = "Literal __CMD1__ und Kommando \\gls{BNC}."
        result = self.masker.mask_latex_commands(source)

        self.assertIn("__CMD1__", result.masked_text)
        self.assertIn("__CMD2__", result.placeholder_map)

    def test_unknown_placeholders_remain_unchanged_on_restore(self) -> None:
        text = "Text mit __CMD999__ Token"
        restored = self.masker.restore_latex_commands(text, {"__CMD1__": "\\\\gls{WBG}"})

        self.assertEqual(text, restored)

    def test_masks_inline_and_display_math_without_data_loss(self) -> None:
        source = (
            "Inline $a^2 + b^2 = c^2$ und Display $$\\int_0^1 x^2 dx$$ "
            "sowie \\[ E = mc^2 \\]."
        )
        result = self.masker.mask_latex_commands(source)
        restored = self.masker.restore_latex_commands(result.masked_text, result.placeholder_map)

        self.assertNotIn("$a^2 + b^2 = c^2$", result.masked_text)
        self.assertNotIn("$$\\int_0^1 x^2 dx$$", result.masked_text)
        self.assertEqual(source, restored)

    def test_masks_math_environment_block(self) -> None:
        source = "Vorher\\n\\begin{equation}E = mc^2\\end{equation}\\nNachher"
        result = self.masker.mask_latex_commands(source)
        restored = self.masker.restore_latex_commands(result.masked_text, result.placeholder_map)

        self.assertNotIn("\\begin{equation}", result.masked_text)
        self.assertEqual(source, restored)

    def test_masks_protected_code_environment(self) -> None:
        source = (
            "Codeblock:\\n"
            "\\begin{lstlisting}\\n"
            "for i in range(3):\\n"
            "    print(i)\\n"
            "\\end{lstlisting}"
        )
        result = self.masker.mask_latex_commands(source)
        restored = self.masker.restore_latex_commands(result.masked_text, result.placeholder_map)

        self.assertNotIn("for i in range(3)", result.masked_text)
        self.assertEqual(source, restored)


if __name__ == "__main__":
    unittest.main()
