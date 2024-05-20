CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

symnmf.o: symnmf.c symnmf.h
	gcc ./symnmf.c -o symnmf -lm $(CFLAGS)

# TODO: add clean
# clean:
    # rm -f *.exe