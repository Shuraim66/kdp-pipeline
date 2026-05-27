"""Puzzle-book pipeline — sibling to the coloring-book pipeline.

The puzzle pipeline shares the cover, metadata, KDP-spec, and Click CLI
infrastructure with the coloring pipeline; only the interior is built
algorithmically (mazelib) rather than from AI-generated images. No Fal.ai
calls are made for the interior — only the cover hero.
"""
