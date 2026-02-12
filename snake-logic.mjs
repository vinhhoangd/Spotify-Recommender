export const DIRECTIONS = {
  UP: { x: 0, y: -1 },
  DOWN: { x: 0, y: 1 },
  LEFT: { x: -1, y: 0 },
  RIGHT: { x: 1, y: 0 },
};

const OPPOSITE = {
  UP: "DOWN",
  DOWN: "UP",
  LEFT: "RIGHT",
  RIGHT: "LEFT",
};

function sameCell(a, b) {
  return a.x === b.x && a.y === b.y;
}

function insideGrid(cell, gridSize) {
  return cell.x >= 0 && cell.y >= 0 && cell.x < gridSize && cell.y < gridSize;
}

function cellKey(cell) {
  return `${cell.x},${cell.y}`;
}

export function placeFood(snake, gridSize, rng = Math.random) {
  const occupied = new Set(snake.map(cellKey));
  const empty = [];

  for (let y = 0; y < gridSize; y += 1) {
    for (let x = 0; x < gridSize; x += 1) {
      const key = `${x},${y}`;
      if (!occupied.has(key)) {
        empty.push({ x, y });
      }
    }
  }

  if (empty.length === 0) {
    return null;
  }

  const sample = rng();
  const raw = Number.isFinite(sample) ? sample : 0;
  const clamped = Math.max(0, Math.min(0.999999, raw));
  const idx = Math.floor(clamped * empty.length);
  return empty[idx];
}

export function createInitialState({ gridSize = 16, rng = Math.random } = {}) {
  const center = Math.floor(gridSize / 2);
  const snake = [
    { x: center, y: center },
    { x: center - 1, y: center },
    { x: center - 2, y: center },
  ];

  return {
    gridSize,
    snake,
    direction: "RIGHT",
    nextDirection: "RIGHT",
    food: placeFood(snake, gridSize, rng),
    score: 0,
    paused: false,
    gameOver: false,
  };
}

export function queueDirection(state, nextDirection) {
  if (!DIRECTIONS[nextDirection]) {
    return state;
  }

  if (OPPOSITE[state.direction] === nextDirection) {
    return state;
  }

  return {
    ...state,
    nextDirection,
  };
}

export function togglePause(state) {
  if (state.gameOver) {
    return state;
  }

  return {
    ...state,
    paused: !state.paused,
  };
}

export function restartState(state, rng = Math.random) {
  return createInitialState({ gridSize: state.gridSize, rng });
}

export function advanceState(state, rng = Math.random) {
  if (state.gameOver || state.paused) {
    return state;
  }

  const vector = DIRECTIONS[state.nextDirection];
  const nextHead = {
    x: state.snake[0].x + vector.x,
    y: state.snake[0].y + vector.y,
  };

  if (!insideGrid(nextHead, state.gridSize)) {
    return {
      ...state,
      gameOver: true,
      direction: state.nextDirection,
    };
  }

  const willGrow = sameCell(nextHead, state.food);
  const bodyToCheck = willGrow ? state.snake : state.snake.slice(0, -1);

  if (bodyToCheck.some((segment) => sameCell(segment, nextHead))) {
    return {
      ...state,
      gameOver: true,
      direction: state.nextDirection,
    };
  }

  const snake = [nextHead, ...state.snake];
  if (!willGrow) {
    snake.pop();
  }

  if (!willGrow) {
    return {
      ...state,
      snake,
      direction: state.nextDirection,
    };
  }

  const food = placeFood(snake, state.gridSize, rng);
  const fullGrid = food === null;

  return {
    ...state,
    snake,
    direction: state.nextDirection,
    food,
    score: state.score + 1,
    gameOver: fullGrid,
  };
}
