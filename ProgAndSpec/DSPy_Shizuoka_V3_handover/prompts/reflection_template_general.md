I provided an assistant with the following instructions to perform a task for me:
```
<curr_param>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<side_info>
```

Your task is to write a new instruction for the assistant. The instruction will be used on many other problems of different kinds and sizes, not only on the examples above.

Read the responses and the feedback, and identify the failure patterns behind them: wrong reading of the input, a method that cannot finish at this size, ignored time limits, missing or wrongly shaped output fields, violated constraints, declared objective values that differ from the solution.

Rules for the new instruction:
1. Write general principles that would have prevented these failures on any problem: how to read the instance, how to choose a method for the problem size, how to stay inside the time limit, how to check the solution against every stated constraint, and how to return a complete solution in the required shape.
2. Do not copy facts that belong to one example (specific ids, numbers, field names, or a rule for one problem) unless the same pattern clearly applies to a whole family of problems; then state it for the family.
3. Merge new points into the existing rules or replace weaker ones. Do not just append. Keep the whole instruction under about 4,000 characters.
4. Keep the contract the current instruction already states (return schema, allowed imports, never return an empty solution) unless the feedback shows it is wrong.

Provide the new instructions within ``` blocks.
