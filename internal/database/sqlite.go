package database

import (
	"fmt"
	"github.com/upper/db/v4"
	"github.com/upper/db/v4/adapter/sqlite"
	"log"
)

// Set the database credentials using the ConnectionURL type provided by the
// adapter.
var settings = sqlite.ConnectionURL{
	Database: "maigo.db",
	Options:  map[string]string{},
}

func ConnectSQLite() {
	// Use Open to access the database.
	sess, err := sqlite.Open(settings)
	if err != nil {
		log.Fatal("Open: ", err)
	}
	defer func(sess db.Session) {
		err := sess.Close()
		if err != nil {
			log.Fatal("Close: ", err)
		}
	}(sess)

	// The settings variable has a String method that builds and returns a valid
	// DSN. This DSN may be different depending on the database you're connecting
	// to.
	fmt.Printf("Connected to %q with DSN:\n\t%q", sess.Name(), settings)
}
