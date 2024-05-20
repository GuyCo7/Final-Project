CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

symnmf.o: symnmf.c symnmf.h
	gcc symnmf.c -o symnmf $(CFLAGS) -lm


# TODO: add clean
# clean:
    # rm -f *.exe