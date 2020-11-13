# How to contribute

Simply make a fork and send a pull request. But before creating a pull request, 
there are just a few guidelines you need to follow:

1. Make a fork and switch to a new branch (don't work on master)
    ```
    git checkout -b new-feature
    ```

2. Make sure all tests are passing with pytest:
    ```
    pytest tests/
    ```

3. Finally, make sure to code using PEP8 conventions and check for
erros using `flake8`.


### Found a bug?

If your code doesn't work and the problem is in the 
our implementation, just submit an issue :-).


