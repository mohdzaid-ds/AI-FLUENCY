

# Prompt Ladder Experiment — FastAPI CRUD To-Do API

## 1. Experiment Overview

### Objective

The goal of this experiment was to measure how individual prompt improvements affect the quality and usefulness of an AI-generated response.

The task selected from the Backend AI Engineering track was:

> Build a CRUD To-Do API using FastAPI.

Instead of creating one detailed prompt immediately, I started with a deliberately weak prompt and improved it through five versions.

Each version added exactly **one prompt-engineering layer**. After every run, I compared the output with the previous version and identified what improved, what still failed, and what should be changed next.

### Prompt layers tested

1. Clearer goal
2. Defined audience
3. Real context
4. Specified output format
5. Constraints

There were **six total runs**:

* Baseline
* Version 1
* Version 2
* Version 3
* Version 4
* Version 5

---

# 2. Baseline — Weak Prompt

## Prompt

> Write backend code for a to-do API.

## Representative Output

Claude produced a fairly complete FastAPI CRUD application with a multi-file structure:

```text
todo_api/
├── main.py
├── models.py
├── schemas.py
├── database.py
└── requirements.txt
```

The response included:

* FastAPI
* SQLite database
* Pydantic schemas
* CRUD endpoints
* `POST /todos`
* `GET /todos`
* `GET /todos/{id}`
* `PUT /todos/{id}`
* `DELETE /todos/{id}`
* Instructions for running the application
* Swagger UI testing instructions

Claude also stated that the endpoints had been tested.

## Analysis

### What changed in the prompt?

Nothing. This was the deliberately weak baseline prompt.

### What improved in the output?

There was no previous version to compare against. However, despite the prompt being very vague, Claude inferred many requirements and produced a relatively complete CRUD API.

### What still failed?

The prompt did not specify:

* Which framework to use
* The required CRUD operations
* The target audience
* The project structure
* The task fields
* Whether a database should be used
* How the response should be structured

As a result, Claude made many design decisions on its own, including choosing FastAPI, SQLite, a multi-file architecture, and specific task fields.

### What would I try next?

The first improvement should be to make the goal more specific by explicitly naming FastAPI and CRUD.

---

# 3. Version 1 — Clearer Goal

## Layer Added

**Clearer goal**

## Prompt

> Build a CRUD To-Do API using FastAPI.

## Representative Output

Claude responded that it had built a complete CRUD To-Do API with:

* `POST`
* `GET` list
* `GET` by ID
* `PUT`
* `DELETE`

It also provided:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

and stated that the files were ready to run.

## Analysis

### What changed in the prompt?

The vague request was changed into a specific goal by explicitly naming:

* FastAPI
* CRUD
* To-Do API

### What improved in the output?

The output was more directly aligned with the requested FastAPI CRUD task. However, the practical improvement was relatively small because the baseline response had already inferred FastAPI and CRUD correctly.

This showed that simply making the goal clearer did not always produce a major improvement when the AI had already inferred the intended technology and task.

### What still failed?

The response still did not know:

* Who the API was for
* The user's experience level
* The user's existing project structure
* The exact task fields required
* How the response should be presented

It also relied on the previous conversation context rather than providing a clearly self-contained response.

### What would I try next?

The next improvement should define the intended audience so the AI can adjust the complexity and explanation level.

---

# 4. Version 2 — Defined Audience

## Layer Added

**Defined audience**

## Prompt

> Build a CRUD To-Do API using FastAPI. The API is for a beginner who is learning FastAPI.

## Representative Output

Claude changed its approach significantly.

It decided to make the project beginner-friendly by:

* Keeping the implementation in one file
* Using a Python dictionary instead of a real database
* Adding comments explaining why the code works
* Explaining concepts such as Pydantic models
* Explaining `response_model`
* Explaining path parameters
* Providing step-by-step testing instructions

Claude explicitly described the solution as being designed for a beginner learning FastAPI.

## Analysis

### What changed in the prompt?

A defined audience was added:

> The API is for a beginner who is learning FastAPI.

### What improved in the output?

The output became significantly more beginner-focused.

Instead of producing a multi-file project with database complexity, Claude simplified the implementation to one file and used in-memory storage.

It also focused more on teaching concepts and explaining the reasons behind the code.

### What still failed?

The response still made several implementation decisions without being told what the actual project requirements were.

For example:

* It selected the task fields.
* It selected in-memory storage.
* It selected its own endpoint behavior.
* It did not know the user's existing project structure.

### What would I try next?

The next improvement should provide real context about the existing project and the desired learning workflow.

---

# 5. Version 3 — Real Context

## Layer Added

**Real context**

## Prompt

> Build a CRUD To-Do API using FastAPI. The API is for a beginner who is learning FastAPI. I am currently building this project in a Python FastAPI application using a `main.py` file, and I want to learn how each CRUD operation works step by step.

## Representative Output

Claude adapted the response to the project context.

Instead of giving the complete API immediately, it started with:

> Step 1: App setup + the "Create" operation

It provided code for the Create operation and explained:

* Pydantic models
* Request body validation
* Server-assigned IDs
* In-memory storage
* `POST /todos`
* `response_model`
* HTTP status code `201`

It then instructed the user to test the Create operation before continuing to the Read operations.

## Analysis

### What changed in the prompt?

Real project context was added.

The prompt explained that:

* The project uses FastAPI.
* The application uses `main.py`.
* The user is learning CRUD.
* The user wants to learn each operation step by step.

### What improved in the output?

The output became much more aligned with the actual learning workflow.

Instead of simply handing over a finished application, Claude broke the work into an incremental learning process.

It started with one operation, explained the code, provided testing instructions, and suggested the next operation only after the first step worked.

### What still failed?

The response still did not specify:

* The exact task fields
* The complete endpoint requirements
* The required response structure
* The final implementation constraints

As a result, Claude continued to make API design decisions independently.

### What would I try next?

The next improvement should control the structure of the AI's response so the code, explanation, testing instructions, and next steps are consistently organized.

---

# 6. Version 4 — Specified Output Format

## Layer Added

**Specified output format**

## Prompt

> Build a CRUD To-Do API using FastAPI. The API is for a beginner who is learning FastAPI. I am currently building this project in a Python FastAPI application using a `main.py` file, and I want to learn how each CRUD operation works step by step. Structure your response with these sections: Goal, Code, Explanation, How to Run, How to Test, and Next Step.

## Representative Output

Claude followed the requested structure:

```text
## Goal

Build the Create operation...

## Code

[FastAPI code]

## Explanation

[Explanation of the code]

## How to Run

[Installation and Uvicorn commands]

## How to Test

[Testing instructions]

## Next Step

[Read operation]
```

The response included a concrete testing example:

```json
{
  "title": "Learn FastAPI",
  "description": "finish tutorial"
}
```

It also explained the expected `201` response and validation behavior.

## Analysis

### What changed in the prompt?

A specified output format was added:

* Goal
* Code
* Explanation
* How to Run
* How to Test
* Next Step

### What improved in the output?

The response became more organized and easier to follow.

The code, explanation, setup instructions, testing procedure, and next learning step were clearly separated.

This made the response more useful as a practical tutorial rather than just a block of generated code.

### What still failed?

The response was better organized, but the output still focused on only the Create operation rather than implementing the complete CRUD API.

This showed that controlling the output format improved **presentation**, but did not by itself guarantee **completeness**.

### What would I try next?

The next improvement should add explicit constraints defining exactly what the API must contain and how it should be implemented.

---

# 7. Version 5 — Constraints

## Layer Added

**Constraints**

## Prompt

> Build a CRUD To-Do API using FastAPI. The API is for a beginner who is learning FastAPI. I am currently building this project in a Python FastAPI application using a `main.py` file, and I want to learn how each CRUD operation works step by step. Structure your response with these sections: Goal, Code, Explanation, How to Run, How to Test, and Next Step. The API must support creating, listing, retrieving, updating, and deleting tasks. Each task must have an integer id, a title, and a boolean completed field. Keep the implementation in one `main.py` file and use in-memory storage instead of a database.

## Representative Output

Claude produced the complete CRUD implementation in one `main.py` file.

The implementation included:

```text
POST   /todos
GET    /todos
GET    /todos/{todo_id}
PUT    /todos/{todo_id}
DELETE /todos/{todo_id}
```

Each task contained:

```text
id
title
completed
```

The application used:

```python
fake_db: dict[int, Todo] = {}
```

for in-memory storage.

Claude also provided an end-to-end testing sequence:

1. Create a task
2. List tasks
3. Retrieve a task
4. Update a task
5. Delete a task
6. Confirm that the deleted task returns `404`

## Analysis

### What changed in the prompt?

Specific implementation constraints were added:

* The API must support all CRUD operations.
* Each task must have an integer `id`.
* Each task must have a `title`.
* Each task must have a boolean `completed` field.
* The implementation must remain in one `main.py`.
* The API must use in-memory storage instead of a database.

### What improved in the output?

The output became much more aligned with the intended project.

Unlike earlier versions that focused on only one operation, Claude produced all five CRUD operations.

The task structure was also clearly defined, the implementation stayed in one file, and the response provided an end-to-end testing sequence.

### What still failed?

Although the constraints made the implementation much more precise, the response still included some unnecessary explanation and continued to make assumptions about how the API should be taught and tested.

The prompt also did not explicitly require the AI to verify the implementation before considering the solution complete.

### What would I try next?

For a production-quality prompt, I would add verification requirements so the AI must check that each endpoint works and confirm the expected status codes and behavior.

---

# 8. Prompt Ladder Comparison

| Version   | Layer Added      | Main Output Improvement                                                           |
| --------- | ---------------- | --------------------------------------------------------------------------------- |
| Baseline  | None             | Produced a generic but surprisingly complete solution                             |
| Version 1 | Clearer goal     | Explicitly targeted FastAPI and CRUD, but produced limited additional improvement |
| Version 2 | Defined audience | Made the solution significantly more beginner-friendly                            |
| Version 3 | Real context     | Adapted the response to the user's `main.py` learning workflow                    |
| Version 4 | Output format    | Made the response more structured and easier to follow                            |
| Version 5 | Constraints      | Made the implementation much more precise and complete                            |

## Most Valuable Improvements

The biggest observable improvements came from:

### 1. Defined audience

Specifying that the API was for a beginner changed the complexity and teaching style of the response.

### 2. Real context

Providing information about the existing `main.py` project changed the response from a generic solution into a step-by-step learning workflow.

### 3. Constraints

The constraints produced the largest improvement in implementation precision. Claude finally had enough information to generate the complete CRUD API with the required fields and architecture.

### 4. Output format

The output format significantly improved organization, although it did not by itself improve implementation completeness.

## Least Effective Improvement

The **clearer goal** produced the smallest practical improvement.

The baseline prompt was already interpreted by Claude as a FastAPI CRUD task, so explicitly mentioning FastAPI and CRUD did not dramatically change the output.

This was an important finding because it demonstrated that not every prompt improvement produces a significant performance gain.

---

# 9. Honest Failure / Limitation

One important lesson from the experiment was that adding a prompt layer does not automatically make every part of the output better.

The clearest example was Version 4.

Adding an output format made the response easier to read and follow, but Claude still focused on the Create operation rather than delivering the complete CRUD implementation.

Therefore:

> **Better formatting did not automatically produce a more complete implementation.**

Similarly, Version 1 showed that making the goal clearer produced only a small improvement because the AI had already inferred much of the intended task from the weak baseline.

These results show why prompt engineering should be tested experimentally rather than assuming that every additional instruction will improve the result.

---

# 10. Final Reusable Prompt

The following prompt combines the useful layers discovered during the experiment.

> Build a beginner-friendly CRUD To-Do API using FastAPI.
>
> The user is learning FastAPI and wants to understand how each CRUD operation works rather than receiving unexplained code. Implement the API in a single `main.py` file using in-memory storage; do not use a database.
>
> Each task must contain:
>
> * `id`: integer assigned by the server
> * `title`: required string
> * `completed`: boolean, defaulting to `false`
>
> The API must support:
>
> * `POST /todos` — create a task
> * `GET /todos` — list all tasks
> * `GET /todos/{todo_id}` — retrieve one task
> * `PUT /todos/{todo_id}` — update a task
> * `DELETE /todos/{todo_id}` — delete a task
>
> Return appropriate HTTP status codes and return `404` when a requested task does not exist.
>
> Structure the response using these sections:
>
> 1. Goal
> 2. Code
> 3. Explanation
> 4. How to Run
> 5. How to Test
> 6. Verification
>
> Explain important FastAPI concepts in beginner-friendly language. Keep the implementation simple and do not add features that were not requested.
>
> Before considering the solution complete, verify that all five CRUD operations work correctly and describe the expected result and HTTP status code for each test.

---

# 11. Why the Final Prompt Is Better

The final prompt combines the layers that produced meaningful improvements during the experiment.

### Clear goal

The task is explicitly defined as a FastAPI CRUD To-Do API.

### Defined audience

The AI knows that the response should be suitable for a beginner.

### Real context

The prompt explains the learning objective and expected implementation environment.

### Output format

The response must follow a predictable structure.

### Constraints

The required fields, endpoints, file structure, and storage method are explicitly defined.

### Verification

The AI is instructed to verify the CRUD behavior and expected status codes.

Together, these instructions reduce ambiguity and prevent the AI from making unnecessary architectural decisions.

---

# 12. Conclusion

This experiment demonstrated that prompt quality can be improved systematically by changing one variable at a time.

The initial prompt was extremely short:

> "Write backend code for a to-do API."

Despite being vague, the AI produced a surprisingly complete solution. However, it also made many assumptions about the framework, database, project structure, and task design.

Adding a **clearer goal** produced only a small improvement because the AI had already inferred much of the intended task.

Adding a **defined audience** significantly changed the response by making it more beginner-friendly.

Adding **real context** made the response more relevant to the actual learning workflow.

Adding a **specified output format** improved organization and usability.

Adding **constraints** produced the largest improvement in implementation precision by explicitly defining the required CRUD operations, task fields, project structure, and storage method.

The main lesson from the experiment is:

> **Effective prompt engineering is not about making a prompt longer. It is about adding the right information to remove the specific ambiguity that is causing problems in the current output.**

The final prompt is therefore not simply the longest prompt. It is the result of testing individual improvements and keeping the layers that produced useful changes.
