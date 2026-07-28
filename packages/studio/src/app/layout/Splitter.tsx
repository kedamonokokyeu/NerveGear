import { useCallback, useRef } from 'react';

export interface SplitterProps {
  orientation: 'vertical' | 'horizontal'; // vertical = resizes columns (col-resize)
  /** Called with the pointer delta (px) from drag start. */
  onDrag: (delta: number) => void;
  /** Called once when a drag begins (capture the starting size). */
  onDragStart: () => void;
  /** Double-click resets the neighboring pane to its default size. */
  onReset: () => void;
  disabled?: boolean;
}

/** Draggable divider: drag to resize, double-click to reset. */
export function Splitter({ orientation, onDrag, onDragStart, onReset, disabled }: SplitterProps) {
  const origin = useRef(0);

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (disabled) return;
      e.preventDefault();
      origin.current = orientation === 'vertical' ? e.clientX : e.clientY;
      onDragStart();
      const el = e.currentTarget;
      el.setPointerCapture(e.pointerId);
      el.classList.add('splitter--active');
      document.body.style.cursor = orientation === 'vertical' ? 'col-resize' : 'row-resize';

      const move = (ev: PointerEvent) => {
        const pos = orientation === 'vertical' ? ev.clientX : ev.clientY;
        onDrag(pos - origin.current);
      };
      const up = (ev: PointerEvent) => {
        el.releasePointerCapture(ev.pointerId);
        el.classList.remove('splitter--active');
        document.body.style.cursor = '';
        el.removeEventListener('pointermove', move);
        el.removeEventListener('pointerup', up);
      };
      el.addEventListener('pointermove', move);
      el.addEventListener('pointerup', up);
    },
    [orientation, onDrag, onDragStart, disabled],
  );

  return (
    <div
      className={`splitter splitter--${orientation}${disabled ? ' splitter--disabled' : ''}`}
      onPointerDown={onPointerDown}
      onDoubleClick={disabled ? undefined : onReset}
    />
  );
}
