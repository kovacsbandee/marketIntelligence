"""Reusable Dash/Mantine component factories to reduce layout repetition."""
from __future__ import annotations

import dash_mantine_components as dmc


def tab_panel(
    *,
    value: str,
    content_id: str,
    loading_id: str,
    placeholder: str | None = None,
    paper_style: dict | None = None,
    box_style: dict | None = None,
    paper_padding: int | None = 10,
):
    """Create a TabsPanel with a Paper + LoadingOverlay.

    Args:
        value: The tab value.
        content_id: ID for the paper/content container.
        loading_id: ID for the loading overlay.
        placeholder: Optional dimmed text to show initially.
        paper_style: Optional style dict for the paper.
        box_style: Optional style dict for the wrapper box.
        paper_padding: Padding applied to the Paper (default 10).
    """
    paper_children = [dmc.Text(placeholder, c="dimmed")] if placeholder else None
    paper_kwargs = {"id": content_id}
    if paper_padding is not None:
        paper_kwargs["p"] = paper_padding
    if paper_style:
        paper_kwargs["style"] = paper_style

    return dmc.TabsPanel(
        value=value,
        children=[
            dmc.Box(
                [
                    dmc.Paper(**paper_kwargs, children=paper_children),
                    dmc.LoadingOverlay(id=loading_id, visible=False),
                ],
                style=box_style,
            )
        ],
    )
