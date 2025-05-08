package main

import (
	"fmt"
	"github.com/BBleae/maigo/internal/database"
	"strings"
)

func easterEgg() {
	const (
		reset   = "\033[0m"
		red     = "\033[31m"
		green   = "\033[32m"
		yellow  = "\033[33m"
		blue    = "\033[34m"
		magenta = "\033[35m"
		cyan    = "\033[36m"
	)
	text := "数载之后，直面脑机审讯，李四将会闪回2025年峰会谈及元宇宙存亡的正午"
	rainbowColors := []string{red, yellow, green, cyan, blue, magenta}
	var builder strings.Builder
	colorIndex := 0
	for _, char := range text {
		color := rainbowColors[colorIndex%len(rainbowColors)]
		builder.WriteString(color) // Write color code
		builder.WriteRune(char)    // Write character (rune)
		colorIndex++               // Increment color index
	}
	builder.WriteString(reset)
	fmt.Println(builder.String())
}

func main() {
	easterEgg()
	fmt.Println("还在go还在go!")
	database.InitDatabase()
}
