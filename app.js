import {
  DIRECTIONS,
  advanceState,
  createInitialState,
  queueDirection,
  restartState,
  togglePause,
} from "./snake-logic.mjs";

const GRID_SIZE = 16;
const TICK_MS = 140;

const boardEl = document.getElementById("board");
const scoreEl = document.getElementById("score");
const statusEl = document.getElementById("status");
const pauseBtn = document.getElementById("pause-btn");
const restartBtn = document.getElementById("restart-btn");
const dirButtons = document.querySelectorAll("button[data-dir]");

let state = createInitialState({ gridSize: GRID_SIZE, rng: Math.random });

boardEl.style.gridTemplateColumns = `repeat(${GRID_SIZE}, 1fr)`;
boardEl.style.gridTemplateRows = `repeat(${GRID_SIZE}, 1fr)`;

const cells = [];
for (let i = 0; i < GRID_SIZE * GRID_SIZE; i += 1) {
  const cell = document.createElement("div");
  cell.className = "cell";
  boardEl.appendChild(cell);
  cells.push(cell);
}

function toIndex(pos) {
  return pos.y * GRID_SIZE + pos.x;
}

function render() {
  for (const cell of cells) {
    cell.className = "cell";
  }

  for (const segment of state.snake) {
    const idx = toIndex(segment);
    if (cells[idx]) {
      cells[idx].classList.add("snake");
    }
  }

  if (state.food) {
    const foodIdx = toIndex(state.food);
    if (cells[foodIdx]) {
      cells[foodIdx].classList.add("food");
    }
  }

  scoreEl.textContent = `Score: ${state.score}`;

  if (state.gameOver) {
    statusEl.textContent = state.food ? "Game over" : "You win";
  } else if (state.paused) {
    statusEl.textContent = "Paused";
  } else {
    statusEl.textContent = "Running";
  }

  pauseBtn.textContent = state.paused ? "Resume" : "Pause";
}

function setDirection(direction) {
  state = queueDirection(state, direction);
}

function restartGame() {
  state = restartState(state, Math.random);
  render();
}

const KEY_TO_DIRECTION = {
  ArrowUp: "UP",
  ArrowDown: "DOWN",
  ArrowLeft: "LEFT",
  ArrowRight: "RIGHT",
  w: "UP",
  W: "UP",
  a: "LEFT",
  A: "LEFT",
  s: "DOWN",
  S: "DOWN",
  d: "RIGHT",
  D: "RIGHT",
};

window.addEventListener("keydown", (event) => {
  if (event.key in KEY_TO_DIRECTION) {
    event.preventDefault();
    setDirection(KEY_TO_DIRECTION[event.key]);
    return;
  }

  if (event.key === " " || event.key === "p" || event.key === "P") {
    event.preventDefault();
    state = togglePause(state);
    render();
    return;
  }

  if (event.key === "Enter" || event.key === "r" || event.key === "R") {
    if (state.gameOver) {
      restartGame();
    }
  }
});

for (const button of dirButtons) {
  button.addEventListener("click", () => {
    const dir = button.getAttribute("data-dir");
    if (dir && dir in DIRECTIONS) {
      setDirection(dir);
    }
  });
}

pauseBtn.addEventListener("click", () => {
  state = togglePause(state);
  render();
});

restartBtn.addEventListener("click", () => {
  restartGame();
});

setInterval(() => {
  state = advanceState(state, Math.random);
  render();
}, TICK_MS);

render();
