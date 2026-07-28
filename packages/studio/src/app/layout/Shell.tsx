import { useRef, useState, type ReactNode } from 'react';
import { layout } from '../theme/tokens';
import { CommandBar } from './CommandBar';
import { StatusBar } from './StatusBar';
import { Pane } from './Pane';
import { Splitter } from './Splitter';

export interface ShellProps {
  structure: ReactNode;
  viewport: ReactNode;
  processTree: ReactNode;
  properties: ReactNode;
}

const clampW = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v));

/**
 * The enterprise frame: command bar (44px) / three-pane body / status bar (26px).
 * Left = Structure, center = Viewport, right = Process tree over Properties.
 * Splitters drag to resize, double-click to reset; pane chevrons collapse each
 * pane individually; '»' in either right-pane header collapses the whole right
 * column to a thin rail.
 */
export function Shell({ structure, viewport, processTree, properties }: ShellProps) {
  const [leftW, setLeftW] = useState<number>(layout.left.default);
  const [rightW, setRightW] = useState<number>(layout.right.default);
  const [procFrac, setProcFrac] = useState<number>(layout.processTreeFraction);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [procCollapsed, setProcCollapsed] = useState(false);
  const [propsCollapsed, setPropsCollapsed] = useState(false);

  const dragStartLeft = useRef(leftW);
  const dragStartRight = useRef(rightW);
  const dragStartProc = useRef(procFrac);
  const rightColRef = useRef<HTMLDivElement>(null);

  const collapseColumn = () => setRightCollapsed(true);

  const rightTopStyle = procCollapsed
    ? { flex: '0 0 auto' }
    : propsCollapsed
      ? { flex: '1 1 auto' }
      : { flex: `0 0 calc(${(procFrac * 100).toFixed(2)}% - 3px)` };

  return (
    <div className="shell">
      <CommandBar />

      <div className="shell__body">
        {/* left — Structure / Layers */}
        <div
          className="shell__col shell__col--left"
          style={{ width: leftCollapsed ? undefined : leftW }}
        >
          <Pane
            title="Structure"
            side="left"
            collapsed={leftCollapsed}
            onToggleCollapse={() => setLeftCollapsed((c) => !c)}
          >
            {structure}
          </Pane>
        </div>

        <Splitter
          orientation="vertical"
          disabled={leftCollapsed}
          onDragStart={() => {
            dragStartLeft.current = leftW;
          }}
          onDrag={(d) =>
            setLeftW(clampW(dragStartLeft.current + d, layout.left.min, layout.left.max))
          }
          onReset={() => setLeftW(layout.left.default)}
        />

        {/* center — viewport */}
        <div className="shell__col shell__col--center">{viewport}</div>

        <Splitter
          orientation="vertical"
          disabled={rightCollapsed}
          onDragStart={() => {
            dragStartRight.current = rightW;
          }}
          onDrag={(d) =>
            setRightW(clampW(dragStartRight.current - d, layout.right.min, layout.right.max))
          }
          onReset={() => setRightW(layout.right.default)}
        />

        {/* right — Process tree over Properties */}
        <div
          ref={rightColRef}
          className="shell__col shell__col--right"
          style={{ width: rightCollapsed ? undefined : rightW }}
        >
          {rightCollapsed ? (
            <Pane
              title="Process / Properties"
              side="right"
              collapsed
              onToggleCollapse={() => setRightCollapsed(false)}
            >
              {null}
            </Pane>
          ) : (
            <>
              <div className="shell__right-top" style={rightTopStyle}>
                <Pane
                  title="Process"
                  side="top"
                  collapsed={procCollapsed}
                  onToggleCollapse={() => setProcCollapsed((c) => !c)}
                  onCollapseColumn={collapseColumn}
                >
                  {processTree}
                </Pane>
              </div>

              <Splitter
                orientation="horizontal"
                disabled={procCollapsed || propsCollapsed}
                onDragStart={() => {
                  dragStartProc.current = procFrac;
                }}
                onDrag={(d) => {
                  const h = rightColRef.current?.clientHeight ?? 600;
                  setProcFrac(
                    Math.min(0.8, Math.max(0.15, dragStartProc.current + d / h)),
                  );
                }}
                onReset={() => setProcFrac(layout.processTreeFraction)}
              />

              <div
                className="shell__right-bottom"
                style={propsCollapsed ? { flex: '0 0 auto' } : undefined}
              >
                <Pane
                  title="Properties"
                  side="bottom"
                  collapsed={propsCollapsed}
                  onToggleCollapse={() => setPropsCollapsed((c) => !c)}
                  onCollapseColumn={collapseColumn}
                >
                  {properties}
                </Pane>
              </div>
            </>
          )}
        </div>
      </div>

      <StatusBar />
    </div>
  );
}
