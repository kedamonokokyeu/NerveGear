import type { ReactNode } from 'react';

export interface PaneProps {
  title: string;
  collapsed: boolean;
  onToggleCollapse: () => void;
  /** Which edge the re-open rail sits on when collapsed. */
  side: 'left' | 'right' | 'top' | 'bottom';
  /** Optional: collapse the entire column this pane lives in (renders '»'). */
  onCollapseColumn?: () => void;
  children: ReactNode;
}

const COLLAPSE_CHEVRON: Record<PaneProps['side'], string> = {
  left: '‹',
  right: '›',
  top: '⌃',
  bottom: '⌄',
};

const EXPAND_CHEVRON: Record<PaneProps['side'], string> = {
  left: '›',
  right: '‹',
  top: '⌄',
  bottom: '⌃',
};

/**
 * Titled, collapsible panel container. Header: 11px uppercase tracked muted,
 * with a collapse chevron (and optionally a whole-column '»'). Content scrolls
 * independently, 12px padding. Collapsed = zero size with a thin re-open rail
 * (vertical for left/right panes, horizontal for top/bottom panes).
 */
export function Pane({
  title,
  collapsed,
  onToggleCollapse,
  side,
  onCollapseColumn,
  children,
}: PaneProps) {
  if (collapsed) {
    return (
      <div
        className={`pane-rail pane-rail--${side}`}
        role="button"
        title={`Expand ${title}`}
        onClick={onToggleCollapse}
      >
        <span className="pane-rail__chevron">{EXPAND_CHEVRON[side]}</span>
        <span className="pane-rail__label">{title}</span>
      </div>
    );
  }
  return (
    <section className="pane">
      <header className="pane__header">
        <span className="pane__title">{title}</span>
        <span className="pane__actions">
          {onCollapseColumn ? (
            <button
              className="pane__chevron"
              title="Collapse panel column"
              onClick={onCollapseColumn}
            >
              »
            </button>
          ) : null}
          <button
            className="pane__chevron"
            title={`Collapse ${title}`}
            onClick={onToggleCollapse}
          >
            {COLLAPSE_CHEVRON[side]}
          </button>
        </span>
      </header>
      <div className="pane__content">{children}</div>
    </section>
  );
}
