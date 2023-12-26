# thesis-data-generator

## Opis danych wejściowych
Dane wejściowe są przechowywane w pliku JSON o nazwie "input". Zawierają listę obiektów, opisujących poszczególne wartości parametrów wejściowych generatora danych. Innymi słowy, każdy obiekt ma następującą strukturę:

```json
{
  "n_eff": ...,
  "grating_period": ...,
  "mode_coupling_coef": ...
}
```

Gdzie:
+ *n_eff* - efektywny współczynnik,
+ *grating_period* - okres siatki,
+ *mode_coupling_coef* - współczynnik sprzężenia,

Dodatkowo, zakłada się, że długość siatki jest stała i wynosi `10μm`.

## Wzory
Wszelkie wzory, założenia wykorzystane w generacji danych zostały wzięte z https://empossible.net/wp-content/uploads/2020/01/Lecture-Coupled-Mode-Theory.pdf?fbclid=IwAR0nP0KZ6HjWEc875-HIVLuTUKttmAB_B5cVw5brq-z-gdiNxhbL_sVJTXg (strony 17-20) 
