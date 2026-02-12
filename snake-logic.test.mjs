import test from "node:test";
import assert from "node:assert/strict";

import {
  advanceState,
  createInitialState,
  placeFood,
  queueDirection,
} from "./snake-logic.mjs";

test("moves one tile in current direction", () => {
  const state = createInitialState({ gridSize: 8, rng: () => 0 });
  const next = advanceState(state, () => 0);

  assert.deepEqual(next.snake[0], {
    x: state.snake[0].x + 1,
    y: state.snake[0].y,
  });
  assert.equal(next.score, 0);
});

test("cannot reverse direction instantly", () => {
  const state = createInitialState({ gridSize: 8, rng: () => 0 });
  const reversed = queueDirection(state, "LEFT");

  assert.equal(reversed.nextDirection, "RIGHT");
});

test("grows and increments score after eating food", () => {
  const state = {
    gridSize: 6,
    snake: [
      { x: 2, y: 2 },
      { x: 1, y: 2 },
    ],
    direction: "RIGHT",
    nextDirection: "RIGHT",
    food: { x: 3, y: 2 },
    score: 0,
    paused: false,
    gameOver: false,
  };

  const next = advanceState(state, () => 0);

  assert.equal(next.snake.length, 3);
  assert.equal(next.score, 1);
  assert.notDeepEqual(next.food, next.snake[0]);
});

test("game over on boundary collision", () => {
  const state = {
    gridSize: 6,
    snake: [
      { x: 5, y: 2 },
      { x: 4, y: 2 },
      { x: 3, y: 2 },
    ],
    direction: "RIGHT",
    nextDirection: "RIGHT",
    food: { x: 0, y: 0 },
    score: 0,
    paused: false,
    gameOver: false,
  };

  const next = advanceState(state, () => 0);

  assert.equal(next.gameOver, true);
});

test("game over on self collision", () => {
  const state = {
    gridSize: 7,
    snake: [
      { x: 3, y: 3 },
      { x: 3, y: 4 },
      { x: 2, y: 4 },
      { x: 2, y: 3 },
    ],
    direction: "UP",
    nextDirection: "LEFT",
    food: { x: 6, y: 6 },
    score: 0,
    paused: false,
    gameOver: false,
  };

  const next = advanceState(state, () => 0);

  assert.equal(next.gameOver, true);
});

test("food is placed on an empty tile", () => {
  const snake = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 0, y: 1 },
  ];

  const food = placeFood(snake, 3, () => 0);

  assert.notEqual(food, null);
  assert.deepEqual(food, { x: 2, y: 0 });
});
