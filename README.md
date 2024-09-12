This code example is to demonstrate the bus injection model and branch flow model for optimal power flow.
Only for educational purposes, the code is not optimized for performance.

- 3 bus opf result:
  - Bus injection model:
    ```python
    Bus 1: V=1.0119, theta=0.0313
    Bus 2: V=1.0004, theta=-0.0253
    Bus 3: V=0.9880, theta=-0.0060
    Objective: 0.0003
    ```
  - Branch flow model using line current as variable:
    ``` python
    Bus 1: V=1.0000, theta=0.1162
    Bus 2: V=1.0000, theta=0.0499
    Bus 3: V=1.0000, theta=0.0676
    Objective: 0.0000
    ```
  - Branch flow model using line power as variable:
    ```python
    Bus 1: V=1.0128, theta=0.0299
    Bus 2: V=0.9927, theta=-0.0207
    Bus 3: V=0.9948, theta=-0.0091
    Objective: 0.0002
    ```
    
note: the original ac opf is a nonlinear problem, and the solver will run into local minima
even with the same model but with different formulations, maybe this is the reason that the results will be different.