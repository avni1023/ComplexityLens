int count_decimal_places(int n) {
    if (n < 0) {
        n = -n;
    }
    int places = 0;
    do {
        places++;
        n = n / 10;
    } while (n > 0);
    return places;
}
